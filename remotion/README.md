# Paperli Advertisement - Premium Remotion Video

A stunning professional video advertisement for **Paperli**, the AI-powered research paper analyzer, built with Remotion.

## 🎬 Overview

This Remotion project creates a polished 15-second (460 frames @ 30fps) advertisement that showcases Paperli's complete workflow from paper upload to deployment.

### Scene Breakdown

| # | Scene | Frames | Features |
|---|-------|--------|----------|
| 1 | **Opening** | 75 | Animated beaker emoji, "Paperli" branding, tagline |
| 2 | **Upload** | 60 | PDF upload interface with bounce animation |
| 3 | **Analyze** | 75 | 6-feature grid (Architecture, Technical, Analysis, Datasets, Code, Deploy) |
| 4 | **Generate** | 70 | Code example in terminal with syntax highlighting |
| 5 | **Deploy** | 60 | Download & deployment options comparison |
| 6 | **CTA** | 120 | Call-to-action with features, button, and branding |

**Total Duration**: 460 frames = 15.3 seconds @ 30fps

## 🎨 Brand Integration

This advertisement stays true to Paperli's design system:

- **Logo**: Flask emoji (🔬) integrated throughout
- **Primary Color**: `#111111` (Dark Black)
- **Background**: `#f5f5f7` (Light Gray)
- **Surface**: `#ffffff` (White)
- **Accent**: `#555555` (Medium Gray)
- **Typography**:
  - Headings: Outfit (modern, geometric)
  - Body: DM Sans (clean, readable)
  - Code: JetBrains Mono
- **Style**: Minimalist, clean, professional with smooth animations

## ✨ Animation Features

- **Opening Pulse**: Spring animation with bounce effect
- **Staggered Reveals**: Sequential fade-ins for elements
- **Scale Transitions**: Smooth 0.8→1 scale for entry animations
- **Slide Animations**: Elements slide in from left/right
- **Floating Backgrounds**: Subtle animated dot elements
- **Growing Accents**: Animated horizontal lines
- **Button Scaling**: Interactive button scale-up on CTA screen

## 📦 Repository Structure

```
remotion/
├── PaperliAd.tsx              # Main advertisement composition (850+ lines)
│   ├── OpeningScene          # Logo + branding intro
│   ├── PDFUploadScene        # Upload interface demo
│   ├── AnalysisScene         # 6-feature showcase
│   ├── CodeGenerationScene   # Code generation example
│   ├── DownloadScene         # Deploy/download options
│   ├── CTAScene              # Final call-to-action
│   └── PaperliAdvertisement  # Main composition wrapper
├── Root.tsx                   # Remotion root configuration
├── remotion.config.ts         # Render settings (H.264, AAC)
├── package.json               # Dependencies & build scripts
└── README.md                  # This file
```

## 🚀 Quick Start Guide

### 1. Installation

```bash
cd remotion
npm install
```

### 2. Preview Mode

Watch in browser with interactive scrubber:

```bash
npm start
```

Opens: `http://localhost:3000`

### 3. Render Final Video

Generate MP4:

```bash
npm run build
```

**Output**: `output.mp4` (1920×1080, ~2-5MB, 2-5 min render time)

### 4. Export Still Frame

Get a keyframe image:

```bash
npm run still
```

**Output**: `still.png` (1920×1080)

## 📊 Video Specifications

| Property | Value |
|----------|-------|
| Resolution | 1920×1080 (Full HD) |
| Frame Rate | 30 fps |
| Frames | 460 |
| Duration | 15.3 seconds |
| Codec | H.264 (MP4) |
| Audio | AAC @ 192kbps |
| Color Profile | YUV420p |
| Bit Rate | ~1500-2000 kbps |

## ✏️ Customization

### Change Headline Text

Edit in `CTAScene`:

```typescript
>Analyze Research<br />Like Never Before
```

### Modify Button Text

```typescript
>Get Started Free
```

### Update Website URL

```typescript
>www.paperli.app
```

### Adjust Colors

At top of `PaperliAd.tsx`:

```typescript
const BRAND_COLOR = '#111111';      // Primary
const LIGHT_BG = '#f5f5f7';         // Background
const WHITE = '#ffffff';             // Cards
const ACCENT_GRAY = '#555555';      // Text
```

### Change Animation Timing

In `PaperliAdvertisement` composition:

```typescript
<Sequence from={0} durationInFrames={75}>
  <OpeningScene />
</Sequence>

// Adjust durationInFrames:
// 30 fps × 2 seconds = 60 frames
// 30 fps × 3 seconds = 90 frames
```

### Add Background Music

Place MP3 in `audio/` folder, then import in any scene:

```typescript
import { Audio } from 'remotion';

export const MyScene = () => (
  <Frame>
    <Audio src="/audio/background-music.mp3" />
    {/* Content */}
  </Frame>
);
```

## 🎥 Advanced Rendering

### 4K Output

```bash
remotion render remotion/Root.tsx PaperliAdvertisement output-4k.mp4 \
  --width 3840 --height 2160 --crf 18
```

### 60 FPS Smooth Motion

```bash
remotion render remotion/Root.tsx PaperliAdvertisement output-60fps.mp4 \
  --fps 60
```

### Fast Render (Lower Quality)

```bash
remotion render remotion/Root.tsx PaperliAdvertisement output-fast.mp4 \
  --crf 28 --concurrency 4
```

### Instagram Reels (1080×1920)

```bash
remotion render remotion/Root.tsx PaperliAdvertisement output-reel.mp4 \
  --width 1080 --height 1920
```

### YouTube Shorts (1080×1920)

```bash
remotion render remotion/Root.tsx PaperliAdvertisement output-shorts.mp4 \
  --width 1080 --height 1920 --fps 60
```

## 📤 Distribution Channels

Ready for:
- ✅ YouTube & YouTube Shorts
- ✅ TikTok & Instagram Reels
- ✅ LinkedIn
- ✅ Twitter/X
- ✅ Facebook
- ✅ Website embedding
- ✅ Email marketing
- ✅ Presentations

## 🎯 Scene Details

### 1. Opening (75 frames / 2.5s)

- Animated beaker emoji with spring bounce
- "Paperli" text fades in and slides up
- "Start Your Research" tagline appears
- Accent line grows from center
- Floating background dots animate
- Beautiful fade-in sequence

### 2. Upload (60 frames / 2s)

- Header with logo and "01. UPLOAD" label
- Large upload box slides in from left
- Bouncing PDF emoji
- "Upload PDF" headline
- Descriptive text
- Dashed border design

### 3. Analyze (75 frames / 2.5s)

- Header with logo and "02. AI ANALYSIS" label
- 6-card grid appears in sequence:
  - 🏗️ Architecture
  - ⚙️ Technical
  - 📊 Analysis
  - 💾 Datasets
  - 📝 Code
  - 🚀 Deploy
- Staggered scale-in animations with shadows

### 4. Generate (70 frames / 2.3s)

- Code example displays in terminal window
- Dark theme with syntax highlighting
- Green text on dark background
- "Runnable code · Ready to implement" caption

### 5. Deploy (60 frames / 2s)

- Download and Deploy sections side-by-side
- Icons and text for each option
- Professional layout with spacing

### 6. CTA (120 frames / 4s)

- Logo appears at top
- Large headline: "Analyze Research Like Never Before"
- Subheadline with key benefits
- "Get Started Free" button with spring animation
- 3-feature row (⚡ Instant Analysis, 💾 Download Code, 🚀 Deploy Ready)
- Footer with website and description
- All elements fade in sequentially

## 🏗️ Technical Architecture

### Components

- **Frame**: Wrapper providing consistent background and layout
- **LogoSVG**: (Placeholder for actual SVG if needed)
- **Scenes**: Each scene is a React component returning a Frame

### Animation Patterns

```typescript
// Fade in
opacity: interpolate(frame, [start, end], [0, 1], {
  extrapolateLeft: 'clamp',
  extrapolateRight: 'clamp'
})

// Scale in
transform: `scale(${interpolate(frame, [start, end], [0.8, 1])})`

// Slide in
transform: `translateX(${interpolate(frame, [start, end], [-300, 0])}px)`

// Spring bounce
const scale = spring({ frame, fps, config: { damping: 100 } })
```

### Performance

- Efficient interpolations with clamping to prevent unnecessary calculations
- Minimal component re-renders
- CSS animations for repeated elements (bounce, float)
- No complex calculations in render functions

## 📚 Resources

- [Remotion Website](https://www.remotion.dev/)
- [API Documentation](https://www.remotion.dev/docs/api)
- [Interpolation Guide](https://www.remotion.dev/docs/interpolate)
- [Spring Animations](https://www.remotion.dev/docs/spring)
- [CLI Reference](https://www.remotion.dev/docs/cli)

## 🐛 Troubleshooting

**Video won't render**
- Check Node.js version (14+)
- Ensure ffmpeg is installed
- Try: `npm run build -- --concurrency 1`

**Colors look wrong**
- Verify HEX codes match your brand
- Check color profile (YUV420p required)

**Animation stutters**
- Reduce FPS for preview
- Try full render with `npm run build`

**Preview won't start**
- Port 3000 may be in use
- Check console for errors
- Try: `npx remotion preview remotion/Root.tsx --port 4000`

## 📝 License

Created for the Paperli project. All branding elements are trademarks of Paperli.

---

**Made with ❤️ using Remotion** 🎬

Questions? Visit [remotion.dev](https://www.remotion.dev/) or check the docs!
