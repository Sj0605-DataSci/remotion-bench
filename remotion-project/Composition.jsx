import React from 'react';
import { AbsoluteFill, Sequence, OffthreadVideo, Audio, staticFile, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { wipe } from '@remotion/transitions/wipe';
import { slide } from '@remotion/transitions/slide';
import { clockWipe } from '@remotion/transitions/clock-wipe';
import { flip } from '@remotion/transitions/flip';
import { none } from '@remotion/transitions/none';

// Every one of these is written for real by ToolServer._write_remotion_inputs()
// from the agent's actual tool calls.
const timeline = require('./timeline.json');
const subtitles = require('./subtitles.json');
const textOverlays = require('./text_overlays.json');
const audioTracks = require('./audio_tracks.json');
const config = require('./composition_config.json');

const PRESENTATIONS = { fade, wipe, slide, clockWipe, flip, none };

const positionStyle = (position) => {
  switch (position) {
    case 'top': return { top: 60, bottom: 'auto' };
    case 'lower_third': return { bottom: '30%', top: 'auto' };
    case 'bottom': return { bottom: 60, top: 'auto' };
    default: return { top: '45%', bottom: 'auto' };
  }
};

const Caption = () => {
  const frame = useCurrentFrame();
  const active = subtitles.find((s) => frame >= s.startFrame && frame < s.endFrame);
  if (!active) return null;
  return (
    <div style={{
      position: 'absolute', bottom: 60, width: '100%', textAlign: 'center',
      color: 'white', fontFamily: 'sans-serif', fontSize: 34, fontWeight: 600,
      textShadow: '0 2px 6px rgba(0,0,0,0.9), 0 0 2px rgba(0,0,0,0.9)', padding: '0 40px',
    }}>{active.text}</div>
  );
};

const TextOverlays = () => {
  const frame = useCurrentFrame();
  return textOverlays
    .filter((o) => frame >= o.startFrame && frame < o.endFrame)
    .map((o, i) => (
      <div key={i} style={{
        position: 'absolute', width: '100%', textAlign: 'center', color: 'white',
        fontFamily: 'sans-serif', fontSize: 44, fontWeight: 700, letterSpacing: 2,
        textShadow: '0 2px 8px rgba(0,0,0,0.85)', ...positionStyle(o.position),
      }}>{o.text}</div>
    ));
};

const AudioTracks = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return audioTracks.map((a, i) => {
    const volumeFn = (f) => {
      const t = f / fps;
      if (a.duck_below && a.duck_below.windows.some(([s, e]) => t >= s && t <= e)) {
        return a.duck_below.volume;
      }
      return a.volume;
    };
    return <Audio key={i} src={staticFile(a.file)} volume={volumeFn} />;
  });
};

// Fix for the confirmed reframe bug: the old version wrapped the whole
// composition in a CSS `transform: scale()` guess with no relation to the
// source video's actual intrinsic size vs. the container, so the crop
// window and the visible subject had no real correspondence (0% subject
// retention in testing, even with a correct agent tool call). The right
// primitive for "crop to a new aspect ratio around a focus point" is the
// standard one: object-fit: cover + object-position, applied per-video,
// not a manual transform on the whole tree.
const videoStyle = (reframe) => {
  if (!reframe) return {};
  const { focus_x = 0.5, focus_y = 0.5 } = reframe;
  return {
    width: '100%', height: '100%', objectFit: 'cover',
    objectPosition: `${focus_x * 100}% ${focus_y * 100}%`,
  };
};

export const Reassembly = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: 'black' }}>
      <TransitionSeries>
        {timeline.flatMap((clip, i) => {
          const seq = (
            <TransitionSeries.Sequence key={`clip-${i}`} durationInFrames={clip.durationInFrames}>
              <OffthreadVideo
                src={staticFile(clip.file)}
                startFrom={Math.round((clip.startInSource || 0) * 24)}
                style={videoStyle(config.reframe)}
              />
            </TransitionSeries.Sequence>
          );
          if (clip.transitionOut && i < timeline.length - 1) {
            const presentation = PRESENTATIONS[clip.transitionOut.type] || fade;
            return [
              seq,
              <TransitionSeries.Transition
                key={`t-${i}`}
                presentation={presentation()}
                timing={linearTiming({ durationInFrames: Math.round((clip.transitionOut.duration || 0.5) * 24) })}
              />,
            ];
          }
          return [seq];
        })}
      </TransitionSeries>
      <Caption />
      <TextOverlays />
      <AudioTracks />
    </AbsoluteFill>
  );
};
