import {buildModel, simulate} from './engine.mjs';

self.onmessage = ({data: request}) => {
  try {
    const baseline = buildModel(request.geometry), modified = buildModel(request.geometry, request.scales);
    const config = {...request.config, input: request.geometry.ports[request.config.site]};
    const baseTrace = simulate(baseline, config), modifiedTrace = simulate(modified, config);
    const brief = model => ({n: model.n, geometry: model.geometry, totalArea: model.totalArea, cableVolume: model.cableVolume, somaArea: model.somaArea});
    const transfer = [baseTrace, modifiedTrace].flatMap(trace => ['time', 'frames', 'root', 'a', 'b'].map(key => trace[key].buffer));
    self.postMessage({id: request.id, baseline: brief(baseline), modified: brief(modified), baseTrace, modifiedTrace}, transfer);
  } catch (error) {
    self.postMessage({id: request.id, error: error.message});
  }
};
