import React from 'react';
import { Composition } from 'remotion';
import { PaperliAdvertisement } from './PaperliAd';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="PaperliAdvertisement"
      component={PaperliAdvertisement}
      durationInFrames={460}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{}}
    />
  );
};
