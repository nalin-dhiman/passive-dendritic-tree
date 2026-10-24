# Code and anatomical data notices

## Project code

The JavaScript demonstration and Python implementation are project source code.
No formal redistribution license has been selected for the code. Public visibility
does not itself grant an open-source license. Contact the repository owner about
reuse permissions. NumPy and SciPy retain their respective licenses.

## Allen-derived morphology

The original SWC and reconstructed cortical example come from the **Allen Cell Types
Database**, specimen **588402092**, Allen Institute for Brain Science:

- [Specimen and morphology](https://celltypes.brain-map.org/experiment/morphology/588402092)
- [Exact source SWC](https://api.brain-map.org/api/v2/well_known_file_download/675965556)
- Reconstruction identifier: 675965554; source file identifier: 675965556.
- Original SWC SHA-256:
  `8d3c5fc12fbee445d8cdbe3a2281029e4f87b219689856477e82e60f46d29bb9`.
- Coordinates and radii are in micrometres. Soma and dendritic types 1, 3 and 4
  are retained; the axon is excluded. The derived file preserves the retained
  centerlines, radii, topology and source identifiers.
- The unchanged SWC input is included under `reproduction/allen/data/`.
  Its original header and source tracing metadata are retained. Numerical
  reference arrays retain the same source-use conditions.

Allen-derived geometry and numerical examples are supplied for attributed,
noncommercial educational and research use under the
[Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-of-use)
and [Citation Policy](https://alleninstitute.org/legal/citation-policy).
The source owner's rights are retained. No MIT, Creative Commons or public-domain
designation is asserted for these data. Commercial redistribution requires the
appropriate permission from Allen. This demonstration is not endorsed by Allen.

The geometry is an anatomical reconstruction with its own sampling and radius
measurement limitations. The electrical model represents the retained arbor;
it is not a statement of dendritic completeness or calibrated biological volume.

## Local uploads

Local SWC uploads remain on the user's device. Users must retain the source,
attribution and applicable data-use permissions for uploaded data. There are no
third-party browser scripts, external simulation calls or tracking services.
