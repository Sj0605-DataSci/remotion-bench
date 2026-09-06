import { Composition, registerRoot } from 'remotion';
import { Reassembly } from './Composition';

const timeline = require('./timeline.json');
const config = require('./composition_config.json');

const fps = config.fps || 24;
const totalDurationFrames = timeline.reduce((a, c) => a + c.durationInFrames, 0);
const transitionOverlapFrames = timeline.reduce(
  (a, c) => a + (c.transitionOut ? Math.round((c.transitionOut.duration || 0.5) * fps) : 0), 0,
);
const totalFrames = Math.max(1, totalDurationFrames - transitionOverlapFrames);

const Root = () => (
  <Composition
    id="Reassembly"
    component={Reassembly}
    durationInFrames={totalFrames}
    fps={fps}
    width={config.width || 1280}
    height={config.height || 720}
  />
);

registerRoot(Root);
