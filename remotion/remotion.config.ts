import { Config } from '@remotion/cli/config';

Config.setStillImageFormat('png');
Config.setVideoImageFormat('jpeg');
Config.setCrf(18);
Config.setPixelFormat('yuv420p');

Config.setCodec('h264');
Config.setAudioCodec('aac');
Config.setAudioBitrate('192k');
