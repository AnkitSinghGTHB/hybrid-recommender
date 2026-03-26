/* ═══════════════════════════════════════════════════════════════
   Hybrid Recommender — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

const API = '';  // same origin

// ── State ────────────────────────────────────────────────────────
let datasets = [];
let modelsReady = false;
let debounceTimer = null;

// ── DOM refs ─────────────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const statusPill    = $('#status-pill');
const statusText    = $('#status-text');
const uploadZone    = $('#upload-zone');
const fileInput     = $('#file-input');
const uploadProgress = $('#upload-progress');
const datasetList   = $('#dataset-list');
const statsBar      = $('#stats-bar');
const buildBtn      = $('#build-btn');
const searchSection = $('#search-section');
const weightsSection = $('#weights-section');
const recSection    = $('#rec-section');
const searchInput   = $('#search-input');
const searchSpinner = $('#search-spinner');
const searchResults = $('#search-results');
const recGrid       = $('#rec-grid');
const recQuery      = $('#rec-query');
const toastContainer = $('#toast-container');

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupUpload();
    setupSearch();
    setupWeights();
    setupToggle();
    checkStatus();
});

// ── Toast ────────────────────────────────────────────────────────
function toast(msg, type='info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    toastContainer.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 400); }, 3500);
}

// ── Status ───────────────────────────────────────────────────────
async function checkStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();
        updateStatus(data.status === 'ready');
    } catch(e) { /* server not up yet */ }
}

function updateStatus(ready) {
    modelsReady = ready;
    statusText.textContent = ready ? 'Models Ready' : 'No Data';
    statusPill.classList.toggle('ready', ready);
    searchSection.classList.toggle('hidden', !ready);
    weightsSection.classList.toggle('hidden', !ready);
    if (!ready) recSection.classList.add('hidden');
    if (ready) searchItems('');
}

// ── Upload ───────────────────────────────────────────────────────
function setupUpload() {
    // Click
    uploadZone.addEventListener('click', () => fileInput.click());

    // File pick
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) uploadFiles(e.target.files);
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    });

    // Build button
    buildBtn.addEventListener('click', buildModels);
}

async function uploadFiles(files) {
    uploadProgress.classList.remove('hidden');

    for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
            const data = await res.json();
            datasets = data.datasets;
            toast(`Loaded: ${file.name}`, 'success');
        } catch(e) {
            toast(`Error: ${e.message}`, 'error');
        }
    }

    uploadProgress.classList.add('hidden');
    renderDatasets();
}

function renderDatasets() {
    datasetList.innerHTML = '';
    if (!datasets.length) {
        statsBar.classList.add('hidden');
        return;
    }
    statsBar.classList.remove('hidden');

    let totalRows = 0;
    datasets.forEach((ds, i) => {
        totalRows += ds.rows;
        const el = document.createElement('div');
        el.className = 'dataset-item';
        el.style.animationDelay = `${i * 0.08}s`;

        const badges = [];
        if (ds.has_reviews) badges.push('<span class="badge badge-positive">Reviews</span>');
        if (ds.has_user_data) badges.push('<span class="badge badge-category">Ratings</span>');
        if (ds.has_behavior) badges.push('<span class="badge badge-neutral">Behavior</span>');

        el.innerHTML = `
            <div class="dataset-item-info">
                <div class="dataset-item-name">${ds.name}</div>
                <div class="dataset-item-meta">${ds.rows.toLocaleString()} rows</div>
                <div class="dataset-item-badges">${badges.join('')}</div>
            </div>
            <button class="btn btn-sm btn-danger" data-id="${ds.id}">✕</button>
        `;
        el.querySelector('button').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteDataset(ds.id);
        });
        datasetList.appendChild(el);
    });

    $('#stat-datasets').textContent = datasets.length;
    $('#stat-rows').textContent = totalRows.toLocaleString();
}

async function deleteDataset(id) {
    try {
        await fetch(`${API}/api/datasets/${id}`, { method: 'DELETE' });
        datasets = datasets.filter(d => d.id !== id);
        renderDatasets();
        updateStatus(false);
        toast('Dataset removed', 'info');
    } catch(e) {
        toast('Failed to remove dataset', 'error');
    }
}

async function buildModels() {
    buildBtn.disabled = true;
    buildBtn.innerHTML = '<span class="spinner"></span> Building...';
    toast('Building models — this may take a moment...', 'info');

    try {
        const res = await fetch(`${API}/api/build`, { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Build failed'); }
        const data = await res.json();
        toast(`Models built! ${data.items} items ready.`, 'success');
        $('#stat-items').textContent = data.items;
        updateStatus(true);
    } catch(e) {
        toast(`Build error: ${e.message}`, 'error');
    }

    buildBtn.disabled = false;
    buildBtn.innerHTML = '<span class="btn-icon">⚡</span> Build Models';
}

// ── Search ───────────────────────────────────────────────────────
function setupSearch() {
    searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const q = searchInput.value.trim();
        if (!q && document.activeElement !== searchInput) {
            searchResults.classList.add('hidden');
            searchInput.style.borderBottomLeftRadius = '';
            searchInput.style.borderBottomRightRadius = '';
        }
        searchSpinner.classList.remove('hidden');
        debounceTimer = setTimeout(() => searchItems(q), 350);
    });

    document.addEventListener('click', (e) => {
        if (!searchSection.contains(e.target)) {
            searchResults.classList.add('hidden');
            searchInput.style.borderBottomLeftRadius = '';
            searchInput.style.borderBottomRightRadius = '';
        }
    });

    searchInput.addEventListener('focus', () => {
        if (searchResults.children.length > 0) {
            searchResults.classList.remove('hidden');
            searchInput.style.borderBottomLeftRadius = '0';
            searchInput.style.borderBottomRightRadius = '0';
        }
    });

    searchInput.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const query = searchInput.value.trim();
            if (!query) return;

            // Hide dropdown instantly
            searchResults.classList.add('hidden');
            searchInput.style.borderBottomLeftRadius = '';
            searchInput.style.borderBottomRightRadius = '';

            try {
                // Resolve the best match for the loose query text
                const res = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}&top_n=1`);
                const data = await res.json();
                
                if (data.results && data.results.length > 0 && !data.is_fallback) {
                    const bestMatch = data.results[0].title;
                    searchInput.value = bestMatch; // Update input with exact title
                    getRecommendations(bestMatch); // Fetch recs for the exact title
                } else {
                    // Unmatched query or fallback results returned. 
                    // Search using exactly what they typed.
                    getRecommendations(query);
                }
            } catch(err) {
                toast('Search failed. Please ensure the backend is running.', 'error');
            }
        }
    });
}

async function searchItems(q) {
    try {
        const res = await fetch(`${API}/api/search?q=${encodeURIComponent(q)}&top_n=15`);
        if (!res.ok) throw new Error('Search failed');
        const data = await res.json();
        renderSearchResults(data, q);
    } catch(e) {
        toast(`Search error: ${e.message}`, 'error');
    }
    searchSpinner.classList.add('hidden');
}

function renderSearchResults(data, q) {
    const results = data.results;
    searchResults.innerHTML = '';
    
    if (document.activeElement === searchInput) {
        searchResults.classList.remove('hidden');
        searchInput.style.borderBottomLeftRadius = '0';
        searchInput.style.borderBottomRightRadius = '0';
    }

    if (!results.length) {
        searchResults.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:1rem;">No items found.</p>';
        return;
    }

    if (data.is_fallback && q && q.length > 0) {
        const header = document.createElement('div');
        header.style.padding = '8px 14px';
        header.style.fontSize = '0.75rem';
        header.style.color = 'var(--text-muted)';
        header.style.background = '#f8fafc';
        header.style.fontWeight = '600';
        header.style.textTransform = 'uppercase';
        header.style.borderBottom = '1px solid var(--border)';
        header.textContent = `No exact matches for "${q}". Popular items:`;
        searchResults.appendChild(header);
    }

    results.forEach((item, i) => {
        const sentimentClass = item.avg_sentiment >= 0.05 ? 'positive' : item.avg_sentiment <= -0.05 ? 'negative' : 'neutral';
        const sentimentLabel = sentimentClass === 'positive' ? '👍 Positive' : sentimentClass === 'negative' ? '👎 Negative' : '😐 Neutral';

        const el = document.createElement('div');
        el.className = 'search-item';
        el.innerHTML = `
            <div class="search-item-info">
                <div class="search-item-title">${item.title}</div>
                <div class="search-item-desc">${item.category ? `<span class="badge badge-category">${item.category}</span> ` : ''}${item.description || ''}</div>
            </div>
            <div class="search-item-scores">
                ${renderStars(item.rating)}
                <span class="badge badge-${sentimentClass}">${sentimentLabel}</span>
            </div>
        `;
        el.addEventListener('click', () => {
            searchInput.value = item.title;
            searchResults.classList.add('hidden');
            searchInput.style.borderBottomLeftRadius = '';
            searchInput.style.borderBottomRightRadius = '';
            getRecommendations(item.title);
        });
        searchResults.appendChild(el);
    });
}

// ── Recommendations ──────────────────────────────────────────────
async function getRecommendations(title) {
    recSection.classList.remove('hidden');
    recQuery.textContent = `for "${title}"`;
    recGrid.innerHTML = '<div style="text-align:center;padding:2rem;"><div class="spinner" style="margin:0 auto;"></div></div>';
    recSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const res = await fetch(`${API}/api/recommend/${encodeURIComponent(title)}?top_n=12`);
        if (res.status === 404) {
            renderRecommendations([]);
            return;
        }
        if (!res.ok) throw new Error('Recommendation failed');
        const data = await res.json();
        renderRecommendations(data.recommendations);
    } catch(e) {
        recGrid.innerHTML = `<p style="color:var(--danger);text-align:center;padding:2rem;">${e.message}</p>`;
    }
}

function renderRecommendations(recs) {
    recGrid.innerHTML = '';
    if (!recs || recs.length === 0) {
        recGrid.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;grid-column:1/-1;">No recommendations found matching your query.</p>';
        return;
    }
    recs.forEach((rec, i) => {
        const sentimentClass = rec.sentiment_score >= 0.525 ? 'positive' : rec.sentiment_score <= 0.475 ? 'negative' : 'neutral';
        const sentimentEmoji = sentimentClass === 'positive' ? '👍' : sentimentClass === 'negative' ? '👎' : '😐';

        let reviewsHtml = '';
        if (rec.top_reviews && rec.top_reviews.length > 0) {
            reviewsHtml = '<div class="rec-card-reviews">';
            rec.top_reviews.forEach(r => {
                reviewsHtml += `<div class="rec-review-item">"${r}"</div>`;
            });
            reviewsHtml += '</div>';
        }

        const card = document.createElement('div');
        card.className = 'rec-card';
        card.style.animationDelay = `${i * 0.08}s`;
        card.innerHTML = `
            <div class="rec-card-header">
                <div class="rec-card-rank">#${i + 1} Recommendation</div>
                ${rec.category ? `<span class="badge badge-category">${rec.category}</span>` : ''}
            </div>
            <div class="rec-card-title">${rec.title}</div>
            <div class="rec-card-category">
                ${renderStars(rec.rating)}
                <span class="badge badge-${sentimentClass}">${sentimentEmoji}</span>
            </div>
            <div class="rec-card-desc">${rec.description || ''}</div>
            ${reviewsHtml}
            <div class="rec-card-bottom">
                <div class="score-bars">
                    ${scoreBar('Content', rec.content_score, 'content')}
                    ${scoreBar('Collab', rec.collab_score, 'collab')}
                    ${scoreBar('Sentiment', rec.sentiment_score, 'sentiment')}
                    ${scoreBar('Hybrid', rec.hybrid_score, 'hybrid')}
                </div>
                <div class="rec-card-footer">
                    <div class="hybrid-label">Hybrid Score</div>
                    <div class="hybrid-big">${(rec.hybrid_score * 100).toFixed(1)}%</div>
                    <button class="btn btn-sm btn-ghost" onclick="getRecommendations('${rec.title.replace(/'/g, "\\'")}')">
                        Find Similar →
                    </button>
                </div>
            </div>
        `;
        recGrid.appendChild(card);
    });
}

function scoreBar(label, value, cls) {
    const pct = Math.round(value * 100);
    return `
        <div class="score-row">
            <span class="score-label">${label}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill ${cls}" style="width: ${pct}%"></div>
            </div>
            <span class="score-value">${pct}%</span>
        </div>
    `;
}

// ── Stars ────────────────────────────────────────────────────────
function renderStars(rating) {
    const full = Math.round(rating || 0);
    let html = '<div class="stars">';
    for (let i = 1; i <= 5; i++) {
        html += `<span class="star ${i <= full ? 'filled' : ''}">★</span>`;
    }
    html += `</div>`;
    return html;
}

// ── Weight Sliders ───────────────────────────────────────────────
function setupWeights() {
    ['alpha', 'beta', 'gamma'].forEach(name => {
        const slider = $(`#weight-${name}`);
        const display = $(`#val-${name}`);
        slider.addEventListener('input', () => {
            display.textContent = (slider.value / 100).toFixed(2);
        });
        slider.addEventListener('change', updateWeights);
    });
}

async function updateWeights() {
    const a = $('#weight-alpha').value / 100;
    const b = $('#weight-beta').value / 100;
    const g = $('#weight-gamma').value / 100;

    try {
        await fetch(`${API}/api/weights`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ alpha: a, beta: b, gamma: g }),
        });
        toast('Weights updated', 'info');

        // If recommendations are visible, re-fetch
        const currentQuery = recQuery.textContent.replace('for "', '').replace('"', '');
        if (currentQuery && !recSection.classList.contains('hidden')) {
            getRecommendations(currentQuery);
        }
    } catch(e) {
        toast('Failed to update weights', 'error');
    }
}

// ── Toggle datasets section ──────────────────────────────────────
function setupToggle() {
    const btn = $('#toggle-datasets');
    const body = $('#dataset-body');
    btn.addEventListener('click', () => {
        body.classList.toggle('hidden');
        btn.textContent = body.classList.contains('hidden') ? '▶' : '▼';
    });
}
