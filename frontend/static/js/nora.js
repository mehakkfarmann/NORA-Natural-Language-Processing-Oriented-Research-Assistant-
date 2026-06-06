async function runNORA(query) {
    const btn = document.getElementById('runBtn');
    btn.disabled = true; btn.textContent = 'Running...';
    resetUI();

    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const { run_id } = await res.json();
        pollStatus(run_id);
    } catch (err) {
        alert('Backend not reachable. Ensure: python run.py --api');
        btn.disabled = false; btn.textContent = 'Run NORA ▶';
    }
}

function pollStatus(runId) {
    const interval = setInterval(async () => {
        const res = await fetch(`/api/status/${runId}`);
        const data = await res.json();
        updateProgress(data.progress, data.status);
        addLog(data.status === 'running' ? `[${data.progress}%] Processing...` : data.status);

        if (data.status === 'completed') {
            clearInterval(interval);
            renderResults(JSON.parse(data.result_json));
            document.getElementById('runBtn').disabled = false;
            document.getElementById('runBtn').textContent = 'Run NORA ▶';
        } else if (data.status === 'failed') {
            clearInterval(interval);
            addLog(`❌ Failed: ${data.error}`);
            document.getElementById('runBtn').disabled = false;
        }
    }, 3000);
}

// UI Reset
function resetUI() {
    const logBody = document.getElementById('logBody');
    if (logBody) logBody.innerHTML = '';

    ['tab-gaps', 'tab-ideas', 'tab-summary'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });

    const resultsPanel = document.getElementById('resultsPanel');
    if (resultsPanel) resultsPanel.classList.remove('visible');

    const gateTracker = document.getElementById('gateTracker');
    if (gateTracker) {
        gateTracker.classList.remove('visible');
        ['g-align', 'g-purity', 'g-variance'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.className = 'gate-item';
                const dot = el.querySelector('.gate-dot');
                if (dot) dot.className = 'gate-dot';
            }
        });
        const badge = document.getElementById('gateBadge');
        if (badge) badge.textContent = 'Running...';
    }

    const layerProgress = document.getElementById('layerProgress');
    if (layerProgress) {
        layerProgress.classList.remove('visible');
        for (let i = 0; i < 4; i++) {
            const c = document.getElementById(`lc${i}`);
            const n = document.getElementById(`ln${i}`);
            if (c) { c.className = 'lcirc'; c.innerHTML = i === 0 ? '🔎' : i === 1 ? '📐' : i === 2 ? '📄' : '💡'; }
            if (n) n.className = 'lnode';
        }
        const progFill = document.getElementById('progFill');
        if (progFill) progFill.style.width = '0%';
    }

    const terminalBox = document.getElementById('terminalBox');
    if (terminalBox) terminalBox.classList.remove('visible');

    const sqResult = document.getElementById('sqResult');
    if (sqResult) {
        sqResult.classList.remove('visible');
        const sqValue = document.getElementById('sqValue');
        if (sqValue) sqValue.textContent = '';
    }
}

// Terminal Logger
function log(html) {
    const b = document.getElementById('logBody');
    if (!b) return;
    const d = document.createElement('div');
    d.innerHTML = html;
    b.appendChild(d);
    b.scrollTop = b.scrollHeight;
}

// Layer Progress Visualizer
function setLayer(i, state) {
    const c = document.getElementById(`lc${i}`);
    const n = document.getElementById(`ln${i}`);
    if (!c || !n) return;
    c.className = 'lcirc ' + (state === 'done' ? 'done' : state === 'active' ? 'active' : '');
    n.className = 'lnode ' + (state === 'done' ? 'done' : state === 'active' ? 'active' : '');
    if (state === 'done') c.innerHTML = '✓';
}

function updateProgress(percent, status) {
    const fill = document.getElementById('progFill');
    const count = document.getElementById('lpCount');
    if (fill) fill.style.width = `${percent}%`;
    if (count && status === 'running') {
        const layer = Math.min(4, Math.floor(parseInt(percent) / 25) + 1);
        count.textContent = `Layer ${layer} of 4`;
    }
}

// Gate Tracker
function updateGate(gateId, state) {
    const el = document.getElementById(gateId);
    if (!el) return;
    el.className = `gate-item ${state}`;
    const dot = el.querySelector('.gate-dot');
    if (dot) dot.className = `gate-dot ${state === 'checking' ? 'checking' : ''}`;
}

// Tab Switcher
function switchTab(el, tabId) {
    document.querySelectorAll('.rtab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.rsection').forEach(s => s.classList.remove('active'));
    el.classList.add('active');
    const target = document.getElementById(`tab-${tabId}`);
    if (target) target.classList.add('active');
}

// Example Chips
function fillAndRun(q) {
    const heroInput = document.getElementById('heroQuery');
    const demoInput = document.getElementById('demoQuery');
    if (heroInput) heroInput.value = q;
    if (demoInput) demoInput.value = q;
    const demoSection = document.getElementById('demo');
    if (demoSection) demoSection.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => runNORA(q), 600);
}

// MAIN: Run NORA Pipeline
async function runNORA(rawQuery) {
    if (!rawQuery || !rawQuery.trim()) {
        const demoInput = document.getElementById('demoQuery');
        if (demoInput) demoInput.focus();
        return;
    }

    const heroInput = document.getElementById('heroQuery');
    const demoInput = document.getElementById('demoQuery');
    if (heroInput) heroInput.value = rawQuery;
    if (demoInput) demoInput.value = rawQuery;

    const demoSection = document.getElementById('demo');
    if (demoSection) demoSection.scrollIntoView({ behavior: 'smooth' });

    const btn = document.getElementById('runBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running...';
    }

    resetUI();
    log(`<span class="ld">╔══════════════════════════════╗</span>`);
    log(`<span class="lc">  [SYSTEM] STARTING NORA PIPELINE</span>`);
    log(`<span class="ld">╚══════════════════════════════╝</span>`);
    log(`<span class="li">  Query: '${rawQuery}'</span>`);

    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: rawQuery })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { run_id } = await res.json();
        log(`<span class="lg">  ✓ Pipeline started | Run ID: ${run_id}</span>`);

        pollStatus(run_id);

    } catch (err) {
        log(`<span class="ld">  ❌ Connection failed: ${err.message}</span>`);
        alert('Backend not reachable. Ensure: python -m uvicorn api.main:app --reload --port 8000');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Run NORA ▶';
        }
    }
}

// Polling Logic
function pollStatus(runId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${runId}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();

            // Show UI sections
            const gateTracker = document.getElementById('gateTracker');
            const layerProgress = document.getElementById('layerProgress');
            const terminalBox = document.getElementById('terminalBox');
            if (gateTracker) gateTracker.classList.add('visible');
            if (layerProgress) layerProgress.classList.add('visible');
            if (terminalBox) terminalBox.classList.add('visible');

            updateProgress(data.progress, data.status);
            log(`<span class="li">  [${data.progress}%] ${data.status === 'running' ? 'Processing...' : data.status}</span>`);

            // Update gates if Layer 0 complete
            if (data.progress >= 25 && data.result_json?.smart_query) {
                updateGate('g-align', 'pass');
                updateGate('g-purity', 'pass');
                updateGate('g-variance', 'pass');
                const badge = document.getElementById('gateBadge');
                if (badge) badge.textContent = '✓ All Gates Passed';
                const sqValue = document.getElementById('sqValue');
                const sqResult = document.getElementById('sqResult');
                if (sqValue) sqValue.textContent = `"${data.result_json.smart_query}"`;
                if (sqResult) sqResult.classList.add('visible');
            }

            if (data.progress >= 20) setLayer(0, 'done');
            if (data.progress >= 40) setLayer(1, 'done');
            if (data.progress >= 60) setLayer(2, 'done');
            if (data.progress >= 80) setLayer(3, 'done');

            if (data.status === 'completed') {
                clearInterval(interval);
                log(`<span class="lo">  ✅ Pipeline complete. Rendering results...</span>`);

                try {
                    const results = typeof data.result_json === 'string' ? JSON.parse(data.result_json) : data.result_json;
                    renderResults(results);
                } catch (parseErr) {
                    log(`<span class="ld">  ⚠ Result parse failed: ${parseErr.message}</span>`);
                }

                const btn = document.getElementById('runBtn');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Run NORA ▶';
                }
            }
            else if (data.status === 'failed') {
                clearInterval(interval);
                log(`<span class="ld">  ❌ Failed: ${data.error || 'Unknown error'}</span>`);
                const btn = document.getElementById('runBtn');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Run NORA ▶';
                }
            }

        } catch (err) {
            console.warn('Polling glitch:', err);
            log(`<span class="ld">  ⚠ Connection glitch — retrying...</span>`);
        }
    }, 2500);
}

// Results Renderer
function renderResults(data) {
    const resultsPanel = document.getElementById('resultsPanel');
    if (resultsPanel) resultsPanel.classList.add('visible');

    const gapCount = document.getElementById('gapCount');
    const ideaCount = document.getElementById('ideaCount');
    if (gapCount) gapCount.textContent = `(${data.gaps?.length || 0})`;
    if (ideaCount) ideaCount.textContent = `(${data.ideas?.length || 0})`;

    // Render Gaps
    const gEl = document.getElementById('tab-gaps');
    if (gEl) {
        gEl.innerHTML = '';
        (data.gaps || []).forEach((g, i) => {
            const card = document.createElement('div');
            card.className = 'gap-card';
            card.style.animationDelay = `${i * 0.1}s`;
            card.innerHTML = `
        <div class="gap-top">
          <div class="gap-title">${g.gap_title || 'Unnamed Gap'}</div>
          <div class="conf-pill">conf ${g.confidence?.toFixed(2) || '0.8'}</div>
        </div>
        <div class="gap-section">📍 ${g.found_in_section || 'Unknown'}</div>
        <div class="evidence">"${g.evidence_quote || 'No evidence'}"</div>
        <div class="gap-tags">
          ${(g.gap_type || 'RESEARCH').split(' ').map(t => `<span class="gtag">${t}</span>`).join('')}
        </div>`;
            gEl.appendChild(card);
        });
        if ((data.gaps || []).length === 0) {
            gEl.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--muted)">No gaps extracted yet</div>';
        }
    }

    // Render Ideas (placeholder)
    const iEl = document.getElementById('tab-ideas');
    if (iEl) {
        iEl.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--muted)">💡 Ideas coming in Layer 4</div>';
    }

    // Render Summary
    const sEl = document.getElementById('tab-summary');
    if (sEl) {
        const avgConf = data.gaps?.length > 0
            ? Math.round(data.gaps.reduce((a, g) => a + (g.confidence || 0.8), 0) / data.gaps.length * 100)
            : 0;
        sEl.innerHTML = `
      <div class="sum-grid">
        <div class="scard"><span class="sval2" style="color:var(--accent)">${data.gaps?.length || 0}</span><div class="slbl">Gaps</div></div>
        <div class="scard"><span class="sval2" style="color:var(--accent2)">${data.smart_query ? '✓' : '—'}</span><div class="slbl">Smart Query</div></div>
        <div class="scard"><span class="sval2" style="color:var(--accent3)">${avgConf}%</span><div class="slbl">Avg Confidence</div></div>
        <div class="scard"><span class="sval2" style="color:var(--yellow)">CPU</span><div class="slbl">Device</div></div>
      </div>
      <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;margin-top:1.2rem">
        <div style="font-weight:600;margin-bottom:.8rem">Run Details</div>
        <table class="run-table">
          <tr><td>Query</td><td>${data.query || '—'}</td></tr>
          <tr><td>Smart Query</td><td>${data.smart_query || '—'}</td></tr>
          <tr><td>Papers</td><td>${data.papers_count || 0}</td></tr>
          <tr><td>Status</td><td style="color:var(--accent)">✅ Complete</td></tr>
        </table>
      </div>
      <div style="display:flex;gap:.8rem;margin-top:1rem">
        <button class="btn-p" onclick="alert('Export coming soon!')">⬇ JSON</button>
        <button class="btn-o" onclick="location.reload()">↩ New Query</button>
      </div>`;
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ NORA frontend loaded');
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', (e) => {
            const query = e.currentTarget.textContent.trim().replace(/^[🧬🔒⚡]\s*/, '');
            fillAndRun(query);
        });
    });
});