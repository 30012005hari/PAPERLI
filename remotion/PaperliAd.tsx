import React from 'react';
import {
  Composition,
  Sequence,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
  Img,
  spring,
  Audio,
  SVG,
} from 'remotion';

const BRAND_COLOR = '#111111';
const LIGHT_BG = '#f5f5f7';
const WHITE = '#ffffff';
const ACCENT_GRAY = '#555555';

// Logo SVG component
export const LogoSVG = () => (
  <svg
    viewBox="0 0 1024 565"
    width="100%"
    height="100%"
    style={{ filter: 'drop-shadow(0 10px 30px rgba(0,0,0,0.1))' }}
  >
    {/* Paperli logo - simplified with key paths */}
    <g transform="translate(300, 150)">
      <path
        d="M0 0 C60.62857143 0 60.62857143 0 72.5625 10.1875 C74 12 74 12 74 14 C74.8971875 14.2475 74.8971875 14.2475 75.8125 14.5 C79.17578505 16.80625261 79.99216301 20.20218009 81 24"
        fill="#111111"
      />
      <circle cx="60" cy="50" r="45" fill="none" stroke="#111111" strokeWidth="3" />
      <text x="60" y="65" textAnchor="middle" fontSize="48" fontWeight="700" fill="#111111">
        🔬
      </text>
    </g>
  </svg>
);

const Frame = ({ children }: { children: React.ReactNode }) => (
  <AbsoluteFill style={{ backgroundColor: LIGHT_BG, overflow: 'hidden' }}>
    {children}
  </AbsoluteFill>
);

// Opening Scene: Logo Animation with Paperli Branding
export const OpeningScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame,
    fps,
    config: { damping: 100, mass: 1 },
    delay: 0,
  });

  const opacity = interpolate(frame, [0, 15, 30], [0, 1, 1]);
  const bounceScale = interpolate(frame, [0, 10, 20, 30], [0, 1.1, 0.95, 1], {
    extrapolateRight: 'clamp',
  });

  return (
    <Frame>
      {/* Animated background dots */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          overflow: 'hidden',
        }}
      >
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              width: '200px',
              height: '200px',
              borderRadius: '50%',
              background: `radial-gradient(circle, rgba(17,17,17,0.1) 0%, transparent 70%)`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${4 + i}s ease-in-out infinite`,
              transform: `translateY(${interpolate(frame, [0, 75], [0, 30])}px)`,
            }}
          />
        ))}
      </div>

      {/* Logo Container */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '60%',
          perspective: '1000px',
          opacity,
        }}
      >
        <div
          style={{
            fontSize: '180px',
            opacity,
            transform: `scale(${bounceScale})`,
            textShadow: '0 20px 60px rgba(0,0,0,0.15)',
            animation: 'pulse 2s ease-in-out forwards',
          }}
        >
          🔬
        </div>
      </div>

      {/* Paperli Text */}
      <div
        style={{
          position: 'absolute',
          top: '55%',
          left: '0',
          right: '0',
          textAlign: 'center',
          opacity: interpolate(frame, [15, 45], [0, 1]),
          fontSize: '72px',
          fontWeight: '800',
          color: BRAND_COLOR,
          fontFamily: 'Outfit',
          letterSpacing: '-2px',
          transform: `translateY(${interpolate(frame, [15, 45], [20, 0])}px)`,
        }}
      >
        Paperli
      </div>

      {/* Tagline */}
      <div
        style={{
          position: 'absolute',
          bottom: '80px',
          left: '0',
          right: '0',
          textAlign: 'center',
          opacity: interpolate(frame, [35, 70], [0, 1]),
          fontSize: '32px',
          color: ACCENT_GRAY,
          fontFamily: 'DM Sans',
          fontWeight: '500',
          letterSpacing: '1px',
          transform: `translateY(${interpolate(frame, [35, 70], [20, 0])}px)`,
        }}
      >
        Start Your Research
      </div>

      {/* Bottom accent line */}
      <div
        style={{
          position: 'absolute',
          bottom: '40px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: `${interpolate(frame, [40, 70], [0, 200])}px`,
          height: '3px',
          background: BRAND_COLOR,
          borderRadius: '2px',
        }}
      />

      <style>{`
        @keyframes pulse {
          0% { transform: scale(0); opacity: 0; }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); opacity: 1; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(30px); }
        }
      `}</style>
    </Frame>
  );
};

// Feature Scene: PDF Upload
export const PDFUploadScene: React.FC = () => {
  const frame = useCurrentFrame();

  const translateX = interpolate(frame, [0, 30], [-300, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <Frame>
      {/* Header with logo */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '80px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 60px',
          borderBottom: `1px solid rgba(0,0,0,0.05)`,
        }}
      >
        <div
          style={{
            fontSize: '32px',
            opacity,
          }}
        >
          🔬
        </div>
        <div
          style={{
            fontSize: '14px',
            fontWeight: '600',
            color: BRAND_COLOR,
            fontFamily: 'Outfit',
            opacity,
            letterSpacing: '0.5px',
          }}
        >
          01. UPLOAD
        </div>
        <div style={{ width: '32px' }} />
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          opacity,
        }}
      >
        <div
          style={{
            width: '320px',
            height: '360px',
            backgroundColor: WHITE,
            borderRadius: '28px',
            border: `3px dashed ${ACCENT_GRAY}`,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '24px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.12)',
            transform: `translateX(${translateX}px)`,
            transition: 'all 0.3s ease',
          }}
        >
          <div style={{ fontSize: '80px', animation: 'bounce 2s infinite' }}>📄</div>
          <div
            style={{
              fontSize: '24px',
              fontWeight: '700',
              color: BRAND_COLOR,
              fontFamily: 'DM Sans',
              textAlign: 'center',
            }}
          >
            Upload PDF
          </div>
          <div
            style={{
              fontSize: '14px',
              color: ACCENT_GRAY,
              textAlign: 'center',
              fontFamily: 'DM Sans',
              maxWidth: '240px',
              lineHeight: '1.5',
            }}
          >
            Drop your research paper here. We'll analyze it instantly.
          </div>
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
      `}</style>
    </Frame>
  );
};

// Feature Scene: AI Analysis
export const AnalysisScene: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const features = [
    { icon: '🏗️', label: 'Architecture' },
    { icon: '⚙️', label: 'Technical' },
    { icon: '📊', label: 'Analysis' },
    { icon: '💾', label: 'Datasets' },
    { icon: '📝', label: 'Code' },
    { icon: '🚀', label: 'Deploy' },
  ];

  return (
    <Frame>
      {/* Header with logo */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '80px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 60px',
          borderBottom: `1px solid rgba(0,0,0,0.05)`,
        }}
      >
        <div
          style={{
            fontSize: '32px',
            opacity,
          }}
        >
          🔬
        </div>
        <div
          style={{
            fontSize: '14px',
            fontWeight: '600',
            color: BRAND_COLOR,
            fontFamily: 'Outfit',
            opacity,
            letterSpacing: '0.5px',
          }}
        >
          02. AI ANALYSIS
        </div>
        <div style={{ width: '32px' }} />
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '24px',
          padding: '120px 60px 60px',
          height: '100%',
          alignContent: 'center',
        }}
      >
        {features.map((feature, index) => {
          const itemOpacity = interpolate(
            frame,
            [10 + index * 7, 20 + index * 7],
            [0, 1],
            { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
          );

          const scale = interpolate(
            frame,
            [10 + index * 7, 20 + index * 7],
            [0.8, 1],
            { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
          );

          return (
            <div
              key={index}
              style={{
                backgroundColor: WHITE,
                borderRadius: '20px',
                padding: '40px 30px',
                textAlign: 'center',
                boxShadow: '0 8px 30px rgba(0,0,0,0.08)',
                opacity: itemOpacity,
                transform: `scale(${scale})`,
                transition: 'all 0.3s ease',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
                justifyContent: 'center',
                alignItems: 'center',
                border: `2px solid ${WHITE}`,
                ':hover': {
                  boxShadow: '0 12px 40px rgba(0,0,0,0.12)',
                },
              }}
            >
              <div style={{ fontSize: '56px' }}>{feature.icon}</div>
              <div
                style={{
                  fontSize: '16px',
                  fontWeight: '700',
                  color: BRAND_COLOR,
                  fontFamily: 'DM Sans',
                  letterSpacing: '0.5px',
                }}
              >
                {feature.label}
              </div>
            </div>
          );
        })}
      </div>
    </Frame>
  );
};

// Feature Scene: Code Generation
export const CodeGenerationScene: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const codeOpacity = interpolate(frame, [10, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const code = `def analyze_paper(pdf_path):
  """AI-powered analysis in seconds"""
  paper = extract_content(pdf_path)
  return generate_insights(paper)`;

  return (
    <Frame>
      <div
        style={{
          position: 'absolute',
          left: '60px',
          top: '60px',
          fontSize: '14px',
          fontWeight: '600',
          color: BRAND_COLOR,
          fontFamily: 'Outfit',
          opacity,
        }}
      >
        03. GENERATE
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          padding: '40px',
        }}
      >
        <div
          style={{
            backgroundColor: '#1e1e2e',
            borderRadius: '16px',
            padding: '24px',
            fontFamily: 'JetBrains Mono',
            fontSize: '14px',
            color: '#a6e3a1',
            maxWidth: '600px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
            opacity: codeOpacity,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: '1.6',
          }}
        >
          {code}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: '40px',
          left: '0',
          right: '0',
          textAlign: 'center',
          fontSize: '18px',
          color: ACCENT_GRAY,
          fontFamily: 'DM Sans',
          opacity: interpolate(frame, [35, 50], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      >
        Runnable code · Ready to implement
      </div>
    </Frame>
  );
};

// Feature Scene: Download & Deploy
export const DownloadScene: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <Frame>
      <div
        style={{
          position: 'absolute',
          left: '60px',
          top: '60px',
          fontSize: '14px',
          fontWeight: '600',
          color: BRAND_COLOR,
          fontFamily: 'Outfit',
          opacity,
        }}
      >
        04. DEPLOY
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          gap: '80px',
        }}
      >
        <div
          style={{
            textAlign: 'center',
            opacity: interpolate(frame, [5, 30], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
          }}
        >
          <div style={{ fontSize: '80px', marginBottom: '20px' }}>📦</div>
          <div
            style={{
              fontSize: '20px',
              fontWeight: '600',
              color: BRAND_COLOR,
              fontFamily: 'DM Sans',
              marginBottom: '8px',
            }}
          >
            Download
          </div>
          <div
            style={{
              fontSize: '14px',
              color: ACCENT_GRAY,
              fontFamily: 'DM Sans',
            }}
          >
            Complete project ZIP
          </div>
        </div>

        <div
          style={{
            textAlign: 'center',
            opacity: interpolate(frame, [10, 35], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
          }}
        >
          <div style={{ fontSize: '80px', marginBottom: '20px' }}>🚀</div>
          <div
            style={{
              fontSize: '20px',
              fontWeight: '600',
              color: BRAND_COLOR,
              fontFamily: 'DM Sans',
              marginBottom: '8px',
            }}
          >
            Deploy
          </div>
          <div
            style={{
              fontSize: '14px',
              color: ACCENT_GRAY,
              fontFamily: 'DM Sans',
            }}
          >
            Optimization guides
          </div>
        </div>
      </div>
    </Frame>
  );
};

// Call to Action Scene
export const CTAScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame,
    fps,
    config: { damping: 80, mass: 0.8 },
    delay: 10,
  });

  const buttonScale = interpolate(frame, [30, 60], [0.8, 1], {
    extrapolateRight: 'clamp',
  });

  return (
    <Frame>
      {/* Background gradient overlay */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `linear-gradient(135deg, ${LIGHT_BG} 0%, ${WHITE} 100%)`,
        }}
      />

      {/* Logo at top */}
      <div
        style={{
          position: 'absolute',
          top: '40px',
          left: '50%',
          transform: 'translateX(-50%)',
          fontSize: '48px',
          opacity: interpolate(frame, [0, 15], [0, 0.8]),
        }}
      >
        🔬
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          gap: '30px',
          padding: '40px',
        }}
      >
        {/* Main Headline */}
        <div
          style={{
            fontSize: '64px',
            fontWeight: '900',
            color: BRAND_COLOR,
            fontFamily: 'Outfit',
            textAlign: 'center',
            letterSpacing: '-2px',
            opacity: interpolate(frame, [0, 20], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
            transform: `translateY(${interpolate(frame, [0, 20], [30, 0])}px)`,
            lineHeight: '1.1',
          }}
        >
          Analyze Research<br />Like Never Before
        </div>

        {/* Subheadline */}
        <div
          style={{
            fontSize: '22px',
            color: ACCENT_GRAY,
            fontFamily: 'DM Sans',
            textAlign: 'center',
            maxWidth: '700px',
            opacity: interpolate(frame, [15, 40], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
            transform: `translateY(${interpolate(frame, [15, 40], [20, 0])}px)`,
            fontWeight: '400',
            lineHeight: '1.6',
          }}
        >
          Powered by AI · Step-by-step analysis · Download & deploy instantly
        </div>

        {/* CTA Button */}
        <div
          style={{
            margin: '20px 0',
            opacity: interpolate(frame, [35, 65], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
            transform: `scale(${buttonScale})`,
          }}
        >
          <div
            style={{
              backgroundColor: BRAND_COLOR,
              color: WHITE,
              padding: '18px 56px',
              borderRadius: '999px',
              fontSize: '20px',
              fontWeight: '700',
              fontFamily: 'DM Sans',
              cursor: 'pointer',
              boxShadow: '0 15px 40px rgba(17,17,17,0.25)',
              border: `2px solid ${BRAND_COLOR}`,
              transition: 'all 0.3s ease',
              whiteSpace: 'nowrap',
            }}
          >
            Get Started Free
          </div>
        </div>

        {/* Features Row */}
        <div
          style={{
            display: 'flex',
            gap: '40px',
            marginTop: '40px',
            opacity: interpolate(frame, [50, 85], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
            transform: `translateY(${interpolate(frame, [50, 85], [20, 0])}px)`,
          }}
        >
          {[
            { icon: '⚡', text: 'Instant Analysis' },
            { icon: '💾', text: 'Download Code' },
            { icon: '🚀', text: 'Deploy Ready' },
          ].map((item, idx) => (
            <div
              key={idx}
              style={{
                textAlign: 'center',
                opacity: interpolate(frame, [50 + idx * 5, 80 + idx * 5], [0, 1], {
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                }),
              }}
            >
              <div style={{ fontSize: '32px', marginBottom: '8px' }}>{item.icon}</div>
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: BRAND_COLOR,
                  fontFamily: 'DM Sans',
                }}
              >
                {item.text}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div
          style={{
            position: 'absolute',
            bottom: '30px',
            left: '0',
            right: '0',
            textAlign: 'center',
            fontSize: '14px',
            color: '#888888',
            fontFamily: 'DM Sans',
            opacity: interpolate(frame, [80, 110], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
          }}
        >
          www.paperli.app · No installation required · Start free today
        </div>
      </div>
    </Frame>
  );
};

// Main Advertisement Composition
export const PaperliAdvertisement: React.FC = () => {
  return (
    <>
      <Sequence from={0} durationInFrames={75}>
        <OpeningScene />
      </Sequence>

      <Sequence from={75} durationInFrames={60}>
        <PDFUploadScene />
      </Sequence>

      <Sequence from={135} durationInFrames={75}>
        <AnalysisScene />
      </Sequence>

      <Sequence from={210} durationInFrames={70}>
        <CodeGenerationScene />
      </Sequence>

      <Sequence from={280} durationInFrames={60}>
        <DownloadScene />
      </Sequence>

      <Sequence from={340} durationInFrames={120}>
        <CTAScene />
      </Sequence>
    </>
  );
};

export const remotionRoot = () => (
  <Composition
    id="PaperliAd"
    component={PaperliAdvertisement}
    durationInFrames={460}
    fps={30}
    width={1920}
    height={1080}
    defaultProps={{}}
  />
);
