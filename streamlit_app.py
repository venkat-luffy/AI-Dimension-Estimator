def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;700;900&display=swap');
    
    :root {
        --primary-bg: #0a0a0a;
        --card-bg: #1a1a1a;
        --glass-bg: rgba(22, 22, 22, 0.85);
        --border-primary: #333333;
        --neon-blue: #00d4ff;
        --neon-purple: #8b5cf6;
        --neon-green: #10b981;
        --neon-orange: #f59e0b;
        --neon-pink: #ec4899;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-neon: linear-gradient(45deg, #00d4ff, #8b5cf6, #10b981, #f59e0b);
        --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    .stApp {
        background: var(--primary-bg);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.04) 0%, transparent 50%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        animation: backgroundPulse 10s ease-in-out infinite alternate;
    }
    
    @keyframes backgroundPulse {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }
    
    /* Enhanced Intro Animation */
    .intro-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        opacity: 1;
        visibility: visible;
        transition: all 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .intro-container.hidden {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
    
    .intro-logo {
        font-size: 5rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        background-size: 400% 400%;
        /* FIXED: Added both webkit and standard background-clip */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: logoAnimation 3s ease-out, gradientFlow 4s ease-in-out infinite;
        transform: translateY(50px);
        opacity: 0;
        animation-fill-mode: both;
    }
    
    @keyframes logoAnimation {
        0% { transform: translateY(50px) scale(0.8); opacity: 0; }
        50% { transform: translateY(0) scale(1.1); opacity: 1; }
        100% { transform: translateY(0) scale(1); opacity: 1; }
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .intro-subtitle {
        font-size: 1.8rem;
        color: var(--neon-blue);
        font-weight: 300;
        text-align: center;
        margin-bottom: 3rem;
        animation: subtitleSlide 2s ease-out 0.5s both;
    }
    
    @keyframes subtitleSlide {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .intro-loader {
        width: 300px;
        height: 6px;
        background: var(--border-primary);
        border-radius: 3px;
        overflow: hidden;
        position: relative;
        animation: loaderAppear 2s ease-out 1s both;
    }
    
    @keyframes loaderAppear {
        0% { transform: scaleX(0); opacity: 0; }
        100% { transform: scaleX(1); opacity: 1; }
    }
    
    .intro-progress {
        height: 100%;
        background: var(--gradient-neon);
        width: 0%;
        transition: width 3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 3px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 0 3rem;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.7) 100%);
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: heroScan 4s linear infinite;
    }
    
    @keyframes heroScan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-title {
        font-size: clamp(3rem, 7vw, 5rem);
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        background-size: 400% 400%;
        /* FIXED: Added both webkit and standard background-clip */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        margin-bottom: 1rem;
        animation: titleGlow 4s ease-in-out infinite, gradientFlow 6s ease-in-out infinite;
        position: relative;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.4)); }
        50% { filter: drop-shadow(0 0 50px rgba(139, 92, 246, 0.6)); }
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: var(--text-secondary);
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Status Grid */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    
    .status-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px); /* Safari compatibility */
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--gradient-neon);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .status-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--neon-blue);
        box-shadow: var(--shadow-glow);
    }
    
    .status-card:hover::before {
        transform: scaleX(1);
    }
    
    .status-header {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .status-icon {
        width: 50px;
        height: 50px;
        background: var(--gradient-primary);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        animation: iconPulse 3s infinite;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes iconPulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(0, 212, 255, 0.5); }
    }
    
    .status-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
    }
    
    .status-value {
        font-weight: 800;
        color: var(--neon-blue);
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
    }
    
    /* Enhanced Navigation Buttons */
    .nav-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2.5rem;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    
    .modern-nav-btn {
        background: var(--card-bg);
        border: 2px solid var(--border-primary);
        border-radius: 24px;
        padding: 3rem 2rem;
        color: var(--text-primary);
        text-decoration: none;
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        cursor: pointer;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); /* Safari compatibility */
    }
    
    .modern-nav-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
        border-radius: 50%;
    }
    
    .modern-nav-btn:hover::before {
        width: 600px;
        height: 600px;
    }
    
    .modern-nav-btn:hover {
        transform: translateY(-12px) rotateX(5deg);
        border-color: var(--neon-blue);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    .nav-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: block;
        animation: iconFloat 4s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0) rotateZ(0deg); }
        50% { transform: translateY(-15px) rotateZ(5deg); }
    }
    
    .nav-title {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Orbitron', monospace;
        margin-bottom: 1rem;
        color: var(--neon-blue);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .nav-description {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Action Buttons */
    .action-btn {
        background: var(--gradient-primary);
        border: none;
        border-radius: 16px;
        padding: 1.2rem 2.5rem;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .action-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
    }
    
    .action-btn:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .action-btn:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }
    
    .action-btn.capture {
        background: linear-gradient(135deg, #f59e0b, #f97316);
    }
    
    .action-btn.detect {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .action-btn.measure {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    }
    
    .action-btn.hologram {
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        position: relative;
    }
    
    .action-btn.hologram::after {
        content: '✨';
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        animation: sparkle 2s infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: translateY(-50%) scale(1); }
        50% { opacity: 0.6; transform: translateY(-50%) scale(1.3); }
    }
    
    /* Content Cards */
    .content-card {
        background: var(--card-bg);
        border: 1px solid var(--border-primary);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1rem;
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); /* Safari compatibility */
        position: relative;
        overflow: hidden;
    }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-neon);
        border-radius: 24px;
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }
    
    .content-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    
    .content-card:hover::before {
        opacity: 1;
    }
    
    /* Interactive Object Selection */
    .detection-image {
        position: relative;
        cursor: crosshair;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .object-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 10;
    }
    
    .object-bbox {
        position: absolute;
        border: 3px solid var(--neon-blue);
        border-radius: 8px;
        background: rgba(0, 212, 255, 0.1);
        cursor: pointer;
        pointer-events: all;
        transition: all 0.3s ease;
        animation: bboxPulse 2s infinite;
    }
    
    @keyframes bboxPulse {
        0%, 100% { box-shadow: 0 0 15px rgba(0, 212, 255, 0.4); }
        50% { box-shadow: 0 0 25px rgba(0, 212, 255, 0.8); }
    }
    
    .object-bbox:hover {
        border-color: var(--neon-green);
        background: rgba(16, 185, 129, 0.2);
        transform: scale(1.05);
    }
    
    .object-bbox.selected {
        border-color: var(--neon-orange);
        background: rgba(245, 158, 11, 0.2);
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.6);
    }
    
    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-neon);
        animation: metricGlow 3s linear infinite;
    }
    
    @keyframes metricGlow {
        0%, 100% { transform: scaleX(0.2); opacity: 0.6; }
        50% { transform: scaleX(1); opacity: 1; }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        border-color: var(--neon-blue);
        box-shadow: var(--shadow-glow);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        /* FIXED: Added both webkit and standard background-clip */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
        animation: valueFlicker 2s infinite;
    }
    
    @keyframes valueFlicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.9rem;
    }
    
    /* Success/Error Messages */
    .success-msg {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 2px solid var(--neon-green);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: var(--neon-green);
        font-weight: 600;
        animation: successPulse 1s ease-out;
    }
    
    @keyframes successPulse {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .error-msg {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Loading States */
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid var(--border-primary);
        border-top: 4px solid var(--neon-blue);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Mission Control Enhancements */
    .mission-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .mission-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); /* Safari compatibility */
    }
    
    .mission-card:hover {
        border-color: var(--neon-blue);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Responsive Design */
    @media (max-width: 1024px) {
        .nav-container {
            grid-template-columns: 1fr;
            gap: 2rem;
        }
        
        .modern-nav-btn {
            padding: 2rem;
        }
        
        .nav-icon {
            font-size: 3rem;
        }
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .intro-logo {
            font-size: 3rem;
        }
        
        .status-grid, .nav-container {
            padding: 1rem;
        }
        
        .content-card {
            padding: 1.5rem;
            margin: 0.5rem;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        
        .mission-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Browser Compatibility Fixes */
    @supports not (backdrop-filter: blur()) {
        .status-card,
        .modern-nav-btn,
        .content-card,
        .mission-card {
            background: var(--card-bg);
        }
    }
    
    /* Firefox compatibility */
    @-moz-document url-prefix() {
        .intro-logo,
        .hero-title,
        .metric-value {
            background: var(--gradient-neon);
            background-clip: text;
            -moz-background-clip: text;
            color: transparent;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Streamlit Overrides */
    .stSelectbox > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div {
        border: 2px dashed var(--neon-blue) !important;
        border-radius: 16px !important;
        background: var(--glass-bg) !important;
    }
    </style>
    """, unsafe_allow_html=True)
