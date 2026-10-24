"""Independent finite-matrix review of saved passive results; no model solves.

Only NumPy and the standard library are used. No cable, stimulus, acquisition,
task, or metric producer is imported. The inverse-based arithmetic here is a
separate check of the saved transposed-solve response-relative calculation.
The conditioning of every matrix is checked before interpreting that check.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def row(x):
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=-1))


def spectral(x):
    return np.linalg.svd(x, compute_uv=False)[..., 0]


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    files = {}
    totals = {"harmonic_arrays": 0, "comparison_arrays": 0, "witness_arrays": 0}
    numbers = {"reference_discrepancy_max": 0., "FVM144_row_error_max": 0.,
               "contraction_ratio_max": 0., "KCL_reconstructed_max": 0.,
               "condition_number_max": 0., "raw_ordering_violation_max": 0.,
               "gain_allocation_max": 0., "response_allocation_max": 0.}
    aggregates = {}
    nominal_root_witness = {}

    def check(name, value, threshold=None):
        passed = bool(value) if threshold is None else bool(value <= threshold)
        checks.append({"check": name, "PASS": passed,
                       "value": value if isinstance(value, (bool, int, float)) else str(value),
                       "maximum_allowed": threshold})

    def close(name, observed, expected, tol=2e-12):
        err = float(np.max(np.abs(observed - expected)))
        scale = max(1., float(np.max(np.abs(expected))))
        check(name, err / scale, tol)

    def read(path):
        path = Path(path)
        files[path.name] = sha256(path)
        return load(path)

    pairs = [(0, 4), (4, 1), (1, 5), (5, 2), (1, 6), (6, 3)]
    paths = sorted(args.raw.glob("harmonic_*.npz"))
    check("35_harmonic_conditions", len(paths) == 35)
    for path in paths:
        key = path.stem.removeprefix("harmonic_")
        h = read(path)
        totals["harmonic_arrays"] += 1
        Z = h["Z"]
        frequency = h["frequency_Hz"]
        check(key + ":shape", Z.shape == (101, 7, 7))
        close(key + ":frequency", frequency, np.arange(101, dtype=float), tol=0.)
        check(key + ":finite", all(np.all(np.isfinite(h[f])) for f in
              ("Z", "reference_Z", "FVM16", "FVM48", "FVM144", "edge_currents")))
        denom = np.maximum(row(h["reference_Z"]), 1.)
        error_ref = float(np.max(row(Z - h["reference_Z"]) / denom))
        numbers["reference_discrepancy_max"] = max(numbers["reference_discrepancy_max"], error_ref)
        check(key + ":continuum_agreement", error_ref, 1e-10)
        close(key + ":reciprocity", Z, Z.swapaxes(-1, -2))
        condition = float(max(np.linalg.cond(h[f]).max() for f in
                         ("Z", "reference_Z", "FVM16", "FVM48", "FVM144")))
        numbers["condition_number_max"] = max(numbers["condition_number_max"], condition)
        check(key + ":conditioning", condition, 1e8)
        H = (Z + Z.conj().swapaxes(-1, -2)) / 2
        passive_min = float(np.min(np.linalg.eigvalsh(H)[..., 0] / spectral(Z)))
        check(key + ":passivity", -passive_min, 1e-10)
        currents = np.zeros((101, 7, 7), dtype=complex)
        for k, (left, right) in enumerate(pairs):
            currents[:, left] += h["edge_currents"][:, k, 0]
            currents[:, right] += h["edge_currents"][:, k, 1]
        kcl = float(np.max(np.abs(currents - np.eye(7))))
        numbers["KCL_reconstructed_max"] = max(numbers["KCL_reconstructed_max"], kcl)
        check(key + ":reconstructed_KCL", kcl, 1e-10)
        errors = [float(np.max(row(h["FVM" + str(n)] - Z) / np.maximum(row(Z), 1.)))
                  for n in (16, 48, 144)]
        ratios = [errors[i + 1] / errors[i] if errors[i] else 0. for i in (0, 1)]
        numbers["FVM144_row_error_max"] = max(numbers["FVM144_row_error_max"], errors[2])
        numbers["contraction_ratio_max"] = max(numbers["contraction_ratio_max"], max(ratios))
        check(key + ":FVM_fine", errors[2], 5e-4)
        check(key + ":FVM_contraction", max(ratios), .35)

        c = read(args.raw / ("comparison_" + key + ".npz"))
        totals["comparison_arrays"] += 1
        B, C = c["baseline_Z"], c["candidate_Z"]
        D = C - B
        b = np.maximum(row(B), 1.)
        A = D / b[..., None]
        # This is intentionally an inverse/multiply calculation, independent
        # of the transposed solve used by the original comparison module.
        R = D @ np.linalg.inv(B)
        grows = row(A)
        root, ports, rel = grows[:, 0], grows.max(axis=1), spectral(R)
        close(key + ":saved_baseline", B, read(args.raw / ("harmonic_baseline_" + key.split("_", 1)[1] + ".npz"))["Z"])
        close(key + ":saved_candidate", C, Z)
        close(key + ":row_scales", c["b_fj"], b)
        close(key + ":normalized_operator", c["A"], A)
        close(key + ":relative_operator", c["R"], R)
        close(key + ":root_metric", c["gain_root_by_frequency"], root)
        close(key + ":all_metric", c["gain_ports_by_frequency"], ports)
        close(key + ":relative_metric", c["response_relative_by_frequency"], rel)
        violation = float(max(np.max(root - ports), np.max(ports - rel), 0.))
        numbers["raw_ordering_violation_max"] = max(numbers["raw_ordering_violation_max"], violation)
        check(key + ":raw_reciprocal_ordering", violation, 2e-12)

        setting = key.split("_", 1)[1]
        baseline_h = read(args.raw / ("harmonic_baseline_" + setting + ".npz"))["FVM144"]
        candidate_h = h["FVM144"]
        Rh = (candidate_h - baseline_h) @ np.linalg.inv(baseline_h)
        alpha_g = 4 * np.max((row(B - baseline_h) + row(C - candidate_h)) / b, axis=1)
        alpha_e = 4 * spectral(R - Rh)
        close(key + ":gain_allocation", c["gain_allocation_by_frequency"], alpha_g)
        close(key + ":relative_allocation", c["response_relative_allocation_by_frequency"], alpha_e)
        numbers["gain_allocation_max"] = max(numbers["gain_allocation_max"], float(alpha_g.max()))
        numbers["response_allocation_max"] = max(numbers["response_allocation_max"], float(alpha_e.max()))
        summary = {"gain_root_raw": float(root.max()), "gain_ports_raw": float(ports.max()),
                   "response_relative_raw": float(rel.max()),
                   "gain_root_upper": float(np.max(root + alpha_g)),
                   "gain_ports_upper": float(np.max(ports + alpha_g)),
                   "response_relative_upper": float(np.max(rel + alpha_e))}
        meta_path = args.processed / ("comparison_" + key + ".json")
        files[meta_path.name] = sha256(meta_path)
        meta = json.loads(meta_path.read_text())
        for field, value in summary.items():
            close(key + ":table_" + field, np.array(meta[field]), np.array(value))
        geometry = key.split("_", 1)[0]
        if geometry not in aggregates:
            aggregates[geometry] = summary
        else:
            for field, value in summary.items():
                aggregates[geometry][field] = max(aggregates[geometry][field], value)

        for kind in ("root_null", "response_relative"):
            w = read(args.raw / ("witness_" + key + "_" + kind + ".npz"))
            totals["witness_arrays"] += 1
            k = int(w["frequency_index"])
            u = w["input_phasor_nA"]
            vb, vc, dv = B[k] @ u, C[k] @ u, D[k] @ u
            close(key + ":" + kind + ":baseline_voltage", w["Vbaseline_phasor_mV"], vb)
            close(key + ":" + kind + ":candidate_voltage", w["Vcandidate_phasor_mV"], vc)
            close(key + ":" + kind + ":difference_voltage", w["Vdelta_phasor_mV"], dv)
            close(key + ":" + kind + ":current_norm", np.array(np.linalg.norm(u)), np.array(.01))
            if kind == "root_null":
                check(key + ":root_null:root_zero", float(abs(dv[0])), 1e-10)
                a = D[k, 0]
                P = np.eye(7) if np.linalg.norm(a) == 0 else np.eye(7) - np.outer(a.conj(), a) / np.linalg.norm(a) ** 2
                close(key + ":projector_hermitian", P, P.conj().T)
                close(key + ":projector_idempotent", P @ P, P)
                j = int(w["selected_output_index"])
                close(key + ":root_null:attainment", np.array(abs(dv[j])), np.array(.01 * np.linalg.norm(D[k, j] @ P)))
                if setting == "nominal" and geometry in ("uniform", "selective"):
                    nominal_root_witness[geometry] = {"frequency_Hz": float(w["frequency_Hz"]),
                        "output_index": j, "root_mV": float(abs(dv[0])), "selected_local_mV": float(abs(dv[j]))}
            else:
                check(key + ":relative:selected_peak", k == int(np.argmax(rel)))
                attained = float(np.linalg.norm(dv) / np.linalg.norm(vb))
                close(key + ":relative:attainment", np.array(attained), np.array(rel[k]))

    output = {"status": "PASS" if all(x["PASS"] for x in checks) else "FAIL",
              "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Saved-matrix arithmetic only. No cable solve, neuron simulation, new input task, or sealed holdout access.",
              "check_count": len(checks), "failed_checks": sum(not x["PASS"] for x in checks),
              "array_counts": totals, "numerical_maxima": numbers,
              "worst_setting_metrics": aggregates, "nominal_root_null_witnesses": nominal_root_witness,
              "source_sha256": files, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in ("status", "check_count", "failed_checks", "array_counts", "numerical_maxima")}))
    raise SystemExit(0 if output["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
