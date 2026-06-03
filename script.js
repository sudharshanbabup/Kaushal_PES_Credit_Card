// ==========================================================================
// 1. DATA AND MODULE CONFIGURATIONS FOR THE ARCHITECTURE BLOCK DIAGRAM
// ==========================================================================
const MODULE_DETAILS = {
    input: {
        title: "<i class='fa-solid fa-file-invoice-dollar text-blue'></i> Module 1: Transaction Input",
        desc: "Captures the transactional data stream at the point-of-sale. The system ingests the standard credit card fields:",
        math: "Features: Time (elapsed seconds), Amount (dollars), and V1–V28 (principal PCA components derived from cardholder characteristics).",
        details: "No patient/cardholder identity is passed directly, maintaining privacy. The dataset exhibits extreme class imbalance (1.7% fraud rate), meaning normal transactions shape 98.3% of the input manifold."
    },
    feature: {
        title: "<i class='fa-solid fa-gears text-blue'></i> Module 2: Feature Engineering",
        desc: "Transforms raw features into robust descriptors and appends attention-supporting interaction terms. Applies a Robust Scaler (using median and IQR) to reduce sensitivity to extreme outliers.",
        math: "Engineered Columns: Hour of Day, Night Indicator, Log Amount, Z-Score Amount, V1/V2 Ratio, Feature Energy (V_magnitude), High Amount Flag, and Amount Rolling Window Deviation.",
        details: "Appends 8 engineered features to the raw 30, expanding the input vector dimension to 38. This increases model capacity to capture nonlinear temporal and value fluctuations."
    },
    aadnn: {
        title: "<i class='fa-solid fa-brain text-orange'></i> Module 3a: Attention-Augmented DNN Branch",
        desc: "The supervised classification branch of the framework. Computes nonlinear feature interactions between disjoint subsets of PCA components utilizing cross-group attention calculations, feeding into a deep Multi-Layer Perceptron (MLP).",
        math: "Architecture: 4 Hidden Layers of dimensions (256, 128, 64, 32) with ReLU activation, outputting a supervised risk probability: p_aadnn ∈ [0, 1].",
        details: "Trained using Binary Cross Entropy Loss on SMOTE-balanced training subsets. It is highly optimized to recognize known fraud patterns but has an inherent blind spot for structurally novel attacks absent from the training set."
    },
    ae: {
        title: "<i class='fa-solid fa-compress text-teal'></i> Module 3b: Autoencoder Anomaly Detector",
        desc: "The unsupervised anomaly detection branch. Trained exclusively on legitimate transactions to learn the normal spending manifold. When fraudulent transactions lie off this manifold, they fail to reconstruct correctly, yielding high error rates.",
        math: "Architecture: Symmetric MLP of dimensions (64, 16, 4, 16, 64) trained to minimize reconstruction Mean Squared Error (MSE). Score: s_ae ∈ [0, 1].",
        details: "By modeling only legitimate behavior, this branch provides robust protection against zero-day or novel fraud types that have never been seen in historical datasets."
    },
    fusion: {
        title: "<i class='fa-solid fa-shuffle text-green'></i> Module 4: Parameterized Risk Scoring",
        desc: "Fuses the supervised probability and the unsupervised anomaly score using a grid-searched parameter alpha (α*) that maximizes validation ROC-AUC.",
        math: "Scoring: s_hybrid = α* · p_aadnn + (1 - α*) · s_ae   (with optimal α* = 0.37)",
        details: "Fusing both branches ensures the system maintains peak performance on standard, known fraud patterns (via AADNN) while retaining the ability to detect emerging, unseen threats (via the Autoencoder)."
    },
    threshold: {
        title: "<i class='fa-solid fa-sliders text-green'></i> Module 5: Dynamic Threshold & SHAP",
        desc: "Rather than using a static decision boundary, the transaction stream is partitioned into temporal windows, and the threshold is adjusted dynamically to optimize F1 score against varying fraud distributions.",
        math: "Dynamic Threshold: τ_t* = argmax_τ [2 · P(τ) · R(τ)] / [P(τ) + R(τ)]  per temporal window.",
        details: "During high-fraud periods (e.g. night hours), the threshold is automatically lowered to prioritize high recall. Local explainability is simultaneously generated using SHAP feature attributions to detail the risk factors."
    },
    output: {
        title: "<i class='fa-solid fa-circle-check text-red'></i> Module 6: Verdict & Attribution",
        desc: "The final system output. Evaluates the hybrid score against the dynamic threshold to classify the transaction.",
        math: "Decision Rule: If s_hybrid ≥ τ_t → Flag as FRAUD (alert risk ops + print SHAP); Else → Approve LEGITIMATE.",
        details: "Flagged transactions are accompanied by a local SHAP chart showing the percentage contribution of the top features, allowing human analysts to instantly verify the system's reasoning."
    }
};

// ==========================================================================
// 2. TAB NAVIGATION LOGIC
// ==========================================================================
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and panes
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to clicked button and corresponding pane
        button.classList.add('active');
        const tabId = button.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// ==========================================================================
// 3. LIGHTBOX AND BLOCK DIAGRAM MODALS LOGIC
// ==========================================================================
// Lightbox
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxCap = document.getElementById('lightbox-caption');
const figureCards = document.querySelectorAll('.figure-card');
const lightboxClose = document.querySelector('.lightbox-close');

figureCards.forEach(card => {
    card.addEventListener('click', () => {
        const figImg = card.querySelector('img');
        lightbox.style.display = 'block';
        lightboxImg.src = figImg.src;
        lightboxCap.innerHTML = `<strong>${card.querySelector('.fig-title').textContent}</strong> - ${card.querySelector('.fig-desc').textContent}`;
    });
});

lightboxClose.addEventListener('click', () => {
    lightbox.style.display = 'none';
});

lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox || e.target === lightboxClose) {
        lightbox.style.display = 'none';
    }
});

// Block Diagram Modals
const modal = document.getElementById('block-modal');
const modalClose = document.querySelector('.modal-close');
const modalBody = document.getElementById('modal-body-content');
const diagramBlocks = document.querySelectorAll('.diagram-block');

diagramBlocks.forEach(block => {
    block.addEventListener('click', () => {
        const moduleKey = block.getAttribute('data-module');
        const data = MODULE_DETAILS[moduleKey];
        
        modalBody.innerHTML = `
            <h3>${data.title}</h3>
            <p>${data.desc}</p>
            <div class="modal-formula">${data.math}</div>
            <p><strong>Detailed Insight:</strong> ${data.details}</p>
        `;
        
        modal.style.display = 'flex';
    });
});

modalClose.addEventListener('click', () => {
    modal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// ==========================================================================
// 4. INTERACTIVE SIMULATOR PLAYGROUND LOGIC
// ==========================================================================
const sliders = {
    amount: document.getElementById('input-amount'),
    energy: document.getElementById('input-energy'),
    v3: document.getElementById('input-v3'),
    hour: document.getElementById('input-hour'),
    ae: document.getElementById('input-ae')
};

const sliderVals = {
    amount: document.getElementById('val-amount'),
    energy: document.getElementById('val-energy'),
    v3: document.getElementById('val-v3'),
    hour: document.getElementById('val-hour'),
    ae: document.getElementById('val-ae')
};

// Interactive scoring parameters
const ALPHA_OPTIMAL = 0.37;
const FIXED_C_FP = 10;

function updateSimulator() {
    // 1. Fetch values from sliders
    const amount = parseFloat(sliders.amount.value);
    const energy = parseFloat(sliders.energy.value);
    const v3 = parseFloat(sliders.v3.value);
    const hour = parseInt(sliders.hour.value);
    const ae_slider = parseFloat(sliders.ae.value);
    
    // 2. Update value text labels
    sliderVals.amount.textContent = `$${amount.toFixed(2)}`;
    sliderVals.energy.textContent = energy.toFixed(2);
    sliderVals.v3.textContent = v3.toFixed(2);
    
    const isNight = (hour >= 22 || hour <= 5);
    sliderVals.hour.textContent = `${hour.toString().padStart(2, '0')}:00 (${isNight ? 'Night' : 'Day'})`;
    sliderVals.ae.textContent = ae_slider.toFixed(2);
    
    // 3. Compute AADNN Probability (Supervised score)
    const logAmount = Math.log1p(amount);
    const nightWeight = isNight ? 0.6 : -0.2;
    // Logit calculation replicating classification pattern
    const z = 0.4 * logAmount - 1.1 * v3 + nightWeight - 1.8;
    const p_aadnn = 1 / (1 + Math.exp(-z));
    
    // 4. Compute Autoencoder reconstruction score
    // final ae score is a combination of the raw slider and feature energy deviation
    const computed_ae = Math.min((energy - 0.5) / 10.0, 1.0);
    const s_ae = Math.max(0, 0.4 * ae_slider + 0.6 * computed_ae);
    
    // 5. Compute Fused Hybrid Score
    const s_hybrid = ALPHA_OPTIMAL * p_aadnn + (1 - ALPHA_OPTIMAL) * s_ae;
    
    // 6. Compute Dynamic Threshold based on temporal window
    // Lower threshold at night to capture more anomalies
    const dynamicThreshold = isNight ? 0.38 : 0.55;
    
    // 7. Render Risk Gauge Visuals
    const gaugeFill = document.querySelector('.gauge-fill');
    const gaugeNeedle = document.querySelector('.gauge-needle');
    const scorePercentLabel = document.getElementById('hybrid-score-percent');
    
    scorePercentLabel.textContent = `${(s_hybrid * 100).toFixed(1)}%`;
    
    // SVG needle rotation: maps [0, 1] to [-90deg, 90deg]
    const needleRotation = -90 + (s_hybrid * 180);
    gaugeNeedle.style.transform = `rotate(${needleRotation}deg)`;
    
    // Gauge Semicircle stroke: circumference of R=40 is ~251.2, semicircle is ~125.6
    const strokeDashoffset = 125.6 * (1 - s_hybrid);
    gaugeFill.style.strokeDashoffset = strokeDashoffset;
    
    // Dynamically color gauge fill based on score
    if (s_hybrid < 0.3) {
        gaugeFill.style.stroke = "var(--accent-green)";
    } else if (s_hybrid < dynamicThreshold) {
        gaugeFill.style.stroke = "var(--accent-orange)";
    } else {
        gaugeFill.style.stroke = "var(--accent-red)";
    }
    
    // 8. Update Classification Verdict Banner
    const verdictAlert = document.getElementById('verdict-alert');
    const verdictTitle = document.getElementById('val-hour'); // Wait, verdict title selector: verdict-title
    const verdictTitleEl = document.getElementById('verdict-title');
    const verdictDescEl = document.getElementById('verdict-description');
    
    // Reset banner classes
    verdictAlert.className = 'verdict-banner';
    
    if (s_hybrid < dynamicThreshold) {
        verdictAlert.classList.add('legit');
        verdictAlert.querySelector('.verdict-icon').innerHTML = '<i class="fa-solid fa-circle-check"></i>';
        verdictTitleEl.textContent = 'Legitimate Transaction';
        verdictDescEl.textContent = `Risk Score (${s_hybrid.toFixed(2)}) is below active threshold (${dynamicThreshold.toFixed(2)}). Approved.`;
    } else if (s_hybrid < dynamicThreshold + 0.15) {
        verdictAlert.classList.add('suspicious');
        verdictAlert.querySelector('.verdict-icon').innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        verdictTitleEl.textContent = 'SUSPICIOUS (Alert Risk Ops)';
        verdictDescEl.textContent = `Score (${s_hybrid.toFixed(2)}) exceeds dynamic threshold (${dynamicThreshold.toFixed(2)}). Sent for review.`;
    } else {
        verdictAlert.classList.add('fraud');
        verdictAlert.querySelector('.verdict-icon').innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
        verdictTitleEl.textContent = 'FRAUD DETECTED (Blocked)';
        verdictDescEl.textContent = `Critical risk score (${s_hybrid.toFixed(2)}) exceeds threshold (${dynamicThreshold.toFixed(2)}). Blocked automatically.`;
    }
    
    // Update Score Pills
    document.getElementById('score-aadnn').textContent = p_aadnn.toFixed(3);
    document.getElementById('score-ae').textContent = s_ae.toFixed(3);
    document.getElementById('score-hybrid').textContent = s_hybrid.toFixed(3);
    document.getElementById('active-threshold').textContent = dynamicThreshold.toFixed(2);
    
    // 9. Generate and Render SHAP Local Explanations
    renderShapAttributions(amount, energy, v3, isNight, p_aadnn, s_ae, s_hybrid, dynamicThreshold);
}

function renderShapAttributions(amount, energy, v3, isNight, p_aadnn, s_ae, s_hybrid, dynamicThreshold) {
    const shapContainer = document.getElementById('shap-bars-container');
    const explanationEl = document.getElementById('human-explanation');
    
    // Compute pseudo-SHAP values for visualization based on physical values
    // Scaling to make contributions sum up to score shift
    const baseValue = 0.10; // Base baseline risk of the dataset
    
    // Calculate raw directional impacts
    const shap_energy = (energy - 1.5) * 0.04;
    const shap_v3 = -v3 * 0.11;
    const shap_amount = (Math.log1p(amount) - 5.0) * 0.03;
    const shap_time = isNight ? 0.12 : -0.04;
    const shap_unseen = (s_ae - 0.05) * 0.15;
    
    const shapValues = [
        { name: 'Feature Energy', val: shap_energy },
        { name: 'PCA Component V3', val: shap_v3 },
        { name: 'Night Transaction', val: shap_time },
        { name: 'Transaction Amount', val: shap_amount },
        { name: 'Autoencoder Manifold', val: shap_unseen }
    ];
    
    // Sort features by absolute contribution
    shapValues.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
    
    // Render Bars
    shapContainer.innerHTML = '';
    shapValues.forEach(f => {
        const isPos = f.val >= 0;
        const absPercent = Math.min(Math.abs(f.val) * 180, 100); // Scale for visual visibility
        
        const row = document.createElement('div');
        row.className = 'shap-row';
        
        const label = document.createElement('div');
        label.className = 'shap-label-text';
        label.textContent = f.name;
        
        const barWrapper = document.createElement('div');
        barWrapper.className = 'shap-bar-wrapper';
        
        const zeroLine = document.createElement('div');
        zeroLine.className = 'shap-zero-line';
        
        const bar = document.createElement('div');
        bar.className = `shap-bar ${isPos ? 'risk-pos' : 'risk-neg'}`;
        
        if (isPos) {
            bar.style.left = '50%';
            bar.style.width = `${absPercent}%`;
        } else {
            bar.style.right = '50%';
            bar.style.width = `${absPercent}%`;
        }
        
        barWrapper.appendChild(zeroLine);
        barWrapper.appendChild(bar);
        row.appendChild(label);
        row.appendChild(barWrapper);
        shapContainer.appendChild(row);
    });
    
    // 10. Generate Human-Readable Description Text
    let explText = "";
    if (s_hybrid < dynamicThreshold) {
        explText = `The transaction of <strong>$${amount.toFixed(2)}</strong> was approved as <strong>LEGITIMATE</strong>. `;
        if (v3 > 0) {
            explText += `The PCA V3 component (${v3.toFixed(2)}) strongly indicates normal baseline spending behaviour. `;
        }
        if (energy < 3.0) {
            explText += `Reconstruction energy remains extremely low (${energy.toFixed(2)}), confirming the transaction maps safely to the normal user profile.`;
        }
    } else {
        explText = `The transaction of <strong>$${amount.toFixed(2)}</strong> was flagged as <strong>FRAUD RISK</strong> because `;
        let reasons = [];
        if (energy > 8.0) {
            reasons.push(`its reconstruction energy (${energy.toFixed(2)}) represents an out-of-distribution anomaly`);
        }
        if (v3 < -1.5) {
            reasons.push(`the PCA V3 value (${v3.toFixed(2)}) matches supervised indicators of historical credit fraud`);
        }
        if (isNight) {
            reasons.push(`it occurred during vulnerable night hours where active thresholds decrease`);
        }
        if (amount > 1500) {
            reasons.push(`the transaction size is large ($${amount.toFixed(2)}), multiplying risk vulnerability`);
        }
        
        if (reasons.length > 0) {
            explText += reasons.join(', and ') + `.`;
        } else {
            explText += `the combined risk sum of AADNN probabilities and reconstruction errors exceeded the dynamic boundary.`;
        }
        
        explText += ` Local SHAP values identify <strong>${shapValues[0].name}</strong> as the primary driver of this alert.`;
    }
    explanationEl.innerHTML = explText;
}

// Bind slider event listeners
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', updateSimulator);
});

// Run initially to populate the dashboard
updateSimulator();
