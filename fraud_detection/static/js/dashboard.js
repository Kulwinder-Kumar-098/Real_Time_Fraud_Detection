/**
 * Real-Time Fraud Operations Dashboard Script
 * Manages live SSE event stream, dynamic KPI updates, Chart.js instances,
 * simulation toggling, and forensic modal inspections.
 */

let timelineChart = null;
let riskDistChart = null;
let eventSource = null;
let isSimulationRunning = false;
let txCountReceived = 0;
const seenTxIds = new Set();

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchInitialData();
  setupLiveStream();

  // Fast live polling interval (2s) ensures transactions ALWAYS render smoothly
  setInterval(pollRecentTransactions, 2000);

  // Periodic metrics & charts refresh (4s)
  setInterval(fetchMetrics, 4000);
});

/* ==========================================================================
   CHARTS INITIALIZATION
   ========================================================================== */

function initCharts() {
  // Chart 1: Volume & Fraud Timeline
  const ctxTimeline = document.getElementById("timelineChart");
  if (ctxTimeline) {
    timelineChart = new Chart(ctxTimeline, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "Approved & Monitored",
            data: [],
            backgroundColor: "#0d9488",
            borderRadius: 3,
            stack: "volume",
          },
          {
            label: "Fraud Flagged",
            data: [],
            backgroundColor: "#dc2626",
            borderRadius: 3,
            stack: "volume",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { font: { family: "Inter", size: 11 }, boxWidth: 12 },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", size: 10 }, maxRotation: 0 },
          },
          y: {
            grid: { color: "#f1f5f9" },
            ticks: { font: { family: "Inter", size: 10 } },
          },
        },
      },
    });
  }

  // Chart 2: Risk Score Distribution
  const ctxRisk = document.getElementById("riskDistChart");
  if (ctxRisk) {
    riskDistChart = new Chart(ctxRisk, {
      type: "bar",
      data: {
        labels: ["0.0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0"],
        datasets: [
          {
            label: "Transaction Count",
            data: [0, 0, 0, 0, 0],
            backgroundColor: [
              "#10b981", // Low risk (green)
              "#34d399", // Mild risk
              "#f59e0b", // Warning (amber)
              "#f97316", // High risk (orange)
              "#dc2626", // Critical (red)
            ],
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", size: 11 } },
          },
          y: {
            grid: { color: "#f1f5f9" },
            ticks: { font: { family: "Inter", size: 10 } },
          },
        },
      },
    });
  }
}

/* ==========================================================================
   METRICS & CHART DATA REFRESH
   ========================================================================== */

async function fetchInitialData() {
  await fetchMetrics();
  await loadInitialTable();
}

async function fetchMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();

    updateKpis(data.kpis);
    updateTimelineChart(data.timeline);
    updateRiskDistChart(data.risk_distribution);
    updateCategoryRiskTable(data.category_risk);
    updateSimulationState(data.simulation);
  } catch (err) {
    console.warn("Metrics fetch error:", err);
  }
}

function updateKpis(kpis) {
  if (!kpis) return;
  document.getElementById("kpi-total-tx").textContent = Number(kpis.total_transactions).toLocaleString();
  document.getElementById("kpi-total-vol").textContent = "$" + Number(kpis.total_volume).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  document.getElementById("kpi-fraud-count").textContent = Number(kpis.fraud_predicted_count).toLocaleString();
  document.getElementById("kpi-fraud-rate").textContent = kpis.predicted_fraud_rate.toFixed(2) + "%";
  document.getElementById("kpi-decline-count").textContent = kpis.decline_count;
  document.getElementById("kpi-review-count").textContent = kpis.review_count;
  document.getElementById("kpi-avg-amount").textContent = "$" + kpis.avg_amount.toFixed(2);
  document.getElementById("kpi-avg-latency").textContent = kpis.avg_latency_ms.toFixed(1) + " ms";

  const latencyPill = document.getElementById("live-latency-pill");
  if (latencyPill) {
    latencyPill.innerHTML = `Latency: <strong>${kpis.avg_latency_ms.toFixed(1)} ms</strong>`;
  }
}

function updateTimelineChart(timeline) {
  if (!timelineChart || !timeline || !timeline.length) return;
  const labels = timeline.map(t => t.time_bucket ? t.time_bucket.split(" ")[1] || t.time_bucket : "");
  const totalCounts = timeline.map(t => Math.max(0, t.total_count - t.fraud_count));
  const fraudCounts = timeline.map(t => t.fraud_count);

  timelineChart.data.labels = labels;
  timelineChart.data.datasets[0].data = totalCounts;
  timelineChart.data.datasets[1].data = fraudCounts;
  timelineChart.update("none");
}

function updateRiskDistChart(dist) {
  if (!riskDistChart || !dist) return;
  const values = [
    dist["0.0 - 0.2"] || 0,
    dist["0.2 - 0.4"] || 0,
    dist["0.4 - 0.6"] || 0,
    dist["0.6 - 0.8"] || 0,
    dist["0.8 - 1.0"] || 0,
  ];
  riskDistChart.data.datasets[0].data = values;
  riskDistChart.update("none");
}

function updateCategoryRiskTable(categories) {
  const tbody = document.getElementById("category-table-body");
  if (!tbody || !categories) return;

  if (categories.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 16px;">No transactions recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = categories.map(cat => {
    let riskBadge = `<span class="badge badge-approve">Low</span>`;
    if (cat.fraud_rate >= 15.0) {
      riskBadge = `<span class="badge badge-decline">Critical</span>`;
    } else if (cat.fraud_rate >= 5.0) {
      riskBadge = `<span class="badge badge-review">Elevated</span>`;
    }

    return `
      <tr>
        <td><strong>${escapeHtml(cat.merchant_category)}</strong></td>
        <td>${Number(cat.total_count).toLocaleString()}</td>
        <td style="color: ${cat.fraud_count > 0 ? 'var(--alert-red)' : 'inherit'}; font-weight: 600;">
          ${cat.fraud_count}
        </td>
        <td style="font-weight: 600;">${cat.fraud_rate.toFixed(1)}%</td>
        <td class="cell-amount">$${Number(cat.avg_amount).toFixed(2)}</td>
        <td>${riskBadge}</td>
      </tr>
    `;
  }).join("");
}

/* ==========================================================================
   LIVE REAL-TIME STREAM (SSE)
   ========================================================================== */

async function loadInitialTable() {
  try {
    const res = await fetch("/api/transactions?limit=15");
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("live-table-body");
    tbody.innerHTML = "";

    if (data.transactions && data.transactions.length > 0) {
      data.transactions.forEach(tx => appendRow(tx, false));
    } else {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 25px;">No transactions found. Start the simulation above.</td></tr>`;
    }
  } catch (err) {
    console.warn("Initial table load error:", err);
  }
}

async function pollRecentTransactions() {
  try {
    const res = await fetch("/api/transactions?limit=15");
    if (!res.ok) return;
    const data = await res.json();
    if (data.transactions && data.transactions.length > 0) {
      const newItems = data.transactions.filter(t => !seenTxIds.has(t.transaction_id));
      if (newItems.length > 0) {
        for (let i = newItems.length - 1; i >= 0; i--) {
          txCountReceived++;
          appendRow(newItems[i], true);
        }
        const tag = document.getElementById("stream-counter-tag");
        if (tag) tag.textContent = `${txCountReceived} transactions processed live`;
      }
    }
  } catch (err) {
    console.warn("Poll recent transactions error:", err);
  }
}

function setupLiveStream() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource("/api/stream");

  eventSource.onopen = () => {
    const tag = document.getElementById("stream-counter-tag");
    if (tag) tag.textContent = "Live Stream Connected";
  };

  eventSource.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "transaction" && msg.payload) {
        txCountReceived++;
        appendRow(msg.payload, true);
        const tag = document.getElementById("stream-counter-tag");
        if (tag) tag.textContent = `${txCountReceived} transactions processed live`;
      }
    } catch (err) {
      console.error("SSE parse error:", err);
    }
  };

  eventSource.onerror = () => {
    // SSE disconnected, fallback polling handles it seamlessly
    const tag = document.getElementById("stream-counter-tag");
    if (tag) tag.textContent = "Live Stream Sync Active";
  };
}

function appendRow(tx, isNew = true) {
  if (!tx || !tx.transaction_id) return;
  if (seenTxIds.has(tx.transaction_id)) return;
  seenTxIds.add(tx.transaction_id);

  const tbody = document.getElementById("live-table-body");
  if (!tbody) return;

  // Clear placeholder row if present
  if (tbody.children.length === 1 && tbody.children[0].innerText.includes("Connecting")) {
    tbody.innerHTML = "";
  }

  const tr = document.createElement("tr");
  if (tx.decision === "DECLINE") {
    tr.classList.add("highlight-fraud");
  }

  // Format decision badge
  let badgeClass = "badge-approve";
  if (tx.decision === "DECLINE") badgeClass = "badge-decline";
  else if (tx.decision === "REVIEW") badgeClass = "badge-review";

  // Risk score pill
  const score = tx.risk_score || 0;
  let riskClass = "low";
  if (score >= 0.80) riskClass = "critical";
  else if (score >= 0.45) riskClass = "high";
  else if (score >= 0.25) riskClass = "medium";

  // Time format
  const timeFormatted = formatTime(tx.timestamp);

  tr.innerHTML = `
    <td class="cell-mono">${timeFormatted}</td>
    <td class="cell-mono"><a href="/transactions/${encodeURIComponent(tx.transaction_id)}" style="color: var(--teal-primary); font-weight: 600; text-decoration: none;">${escapeHtml(tx.transaction_id)}</a></td>
    <td><span style="font-weight: 500;">${escapeHtml(tx.user_id)}</span></td>
    <td>
      <div><strong>${escapeHtml(tx.merchant)}</strong></div>
      <div style="font-size: 10.5px; color: var(--text-muted);">${escapeHtml(tx.merchant_category || 'General')}</div>
    </td>
    <td>${escapeHtml(tx.location)}</td>
    <td style="font-size: 11.5px; color: var(--text-muted); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(tx.device)}">${escapeHtml(tx.device)}</td>
    <td class="cell-amount">$${Number(tx.amount).toFixed(2)}</td>
    <td><span class="risk-pill ${riskClass}">${(score * 100).toFixed(1)}%</span></td>
    <td><span class="badge ${badgeClass}">${tx.decision}</span></td>
    <td>
      <button class="btn btn-secondary btn-sm" onclick="openInspectModal('${escapeHtml(tx.transaction_id)}')">Inspect</button>
    </td>
  `;

  if (isNew) {
    tbody.insertBefore(tr, tbody.firstChild);
    // Keep max 25 rows in live view
    if (tbody.children.length > 25) {
      tbody.removeChild(tbody.lastChild);
    }
  } else {
    tbody.appendChild(tr);
  }
}

/* ==========================================================================
   SIMULATION TOGGLE
   ========================================================================== */

async function toggleSimulation() {
  const btn = document.getElementById("btn-toggle-sim");
  const rateSelect = document.getElementById("sim-rate-select");
  const interval = parseFloat(rateSelect.value || "1.0");

  const endpoint = isSimulationRunning ? "/api/simulation/stop" : "/api/simulation/start";
  const body = isSimulationRunning ? {} : { interval_sec: interval };

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    updateSimulationState(data.details);
    fetchMetrics();
    setTimeout(pollRecentTransactions, 300);
    setTimeout(pollRecentTransactions, 1000);
  } catch (err) {
    console.error("Simulation toggle error:", err);
  }
}

function updateSimulationState(sim) {
  if (!sim) return;
  isSimulationRunning = sim.is_running;

  const btn = document.getElementById("btn-toggle-sim");
  const btnText = document.getElementById("sim-btn-text");
  const badge = document.getElementById("sim-status-badge");
  const bannerDot = document.getElementById("banner-pulse-dot");
  const sideDot = document.getElementById("sidebar-pulse-dot");

  if (isSimulationRunning) {
    if (btn) {
      btn.className = "btn btn-secondary";
      btnText.textContent = "Pause Stream";
    }
    if (badge) {
      badge.className = "badge badge-approve";
      badge.textContent = "STREAM ACTIVE";
    }
    if (bannerDot) bannerDot.className = "pulse-dot running";
    if (sideDot) sideDot.className = "pulse-dot running";
  } else {
    if (btn) {
      btn.className = "btn btn-teal";
      btnText.textContent = "Start Live Stream";
    }
    if (badge) {
      badge.className = "badge badge-review";
      badge.textContent = "STREAM PAUSED";
    }
    if (bannerDot) bannerDot.className = "pulse-dot stopped";
  }
}

/* ==========================================================================
   FORENSIC INSPECTION MODAL
   ========================================================================== */

async function openInspectModal(txId) {
  const modal = document.getElementById("inspect-modal");
  const body = document.getElementById("modal-content-body");
  const title = document.getElementById("modal-tx-title");

  title.textContent = `Forensic Risk Analysis — ${txId}`;
  body.innerHTML = `<div style="text-align: center; padding: 30px; color: var(--text-muted);">Fetching transaction forensic telemetry...</div>`;
  modal.classList.add("open");

  try {
    const res = await fetch(`/api/transactions/${encodeURIComponent(txId)}`);
    if (!res.ok) {
      body.innerHTML = `<div style="color: var(--alert-red); padding: 20px;">Could not retrieve transaction details.</div>`;
      return;
    }
    const tx = await res.json();

    let badgeClass = "badge-approve";
    if (tx.decision === "DECLINE") badgeClass = "badge-decline";
    else if (tx.decision === "REVIEW") badgeClass = "badge-review";

    const reasons = Array.isArray(tx.risk_reasons) ? tx.risk_reasons : [];
    const reasonsHtml = reasons.length > 0
      ? `<div class="risk-reasons-box">
           <div class="risk-reasons-title">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
               <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>
               <line x1="12" y1="8" x2="12" y2="12"/>
               <line x1="12" y1="16" x2="12.01" y2="16"/>
             </svg>
             ML Model Triggered Risk Factors
           </div>
           <ul class="risk-reasons-list">
             ${reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("")}
           </ul>
         </div>`
      : `<div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 10px 14px; border-radius: 4px; font-size: 12px; margin-bottom: 16px;">
           No suspicious anomalies triggered. Transaction aligns with normal user profile.
         </div>`;

    body.innerHTML = `
      ${reasonsHtml}
      <div class="detail-grid">
        <div class="detail-item">
          <div class="detail-key">Transaction ID</div>
          <div class="detail-val cell-mono">${escapeHtml(tx.transaction_id)}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Risk Score / Probability</div>
          <div class="detail-val" style="color: ${tx.risk_score >= 0.42 ? 'var(--alert-red)' : 'var(--teal-primary)'}">
            ${(tx.risk_score * 100).toFixed(2)}% (${tx.risk_tier} RISK)
          </div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Decision</div>
          <div class="detail-val"><span class="badge ${badgeClass}">${tx.decision}</span></div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Amount</div>
          <div class="detail-val cell-amount">$${Number(tx.amount).toFixed(2)}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">User ID</div>
          <div class="detail-val cell-mono">${escapeHtml(tx.user_id)}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Merchant & Category</div>
          <div class="detail-val">${escapeHtml(tx.merchant)} (${escapeHtml(tx.merchant_category)})</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Location</div>
          <div class="detail-val">${escapeHtml(tx.location)}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Device Fingerprint</div>
          <div class="detail-val" style="font-size: 12px;">${escapeHtml(tx.device)}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Payment Method & Channel</div>
          <div class="detail-val">${escapeHtml(tx.payment_method)} • ${escapeHtml(tx.channel || 'Online')}</div>
        </div>
        <div class="detail-item">
          <div class="detail-key">Inference Latency</div>
          <div class="detail-val">${tx.latency_ms || 8.2} ms</div>
        </div>
      </div>
      <div style="text-align: right; margin-top: 14px;">
        <a href="/transactions/${encodeURIComponent(tx.transaction_id)}" class="btn btn-teal btn-sm">Full Forensic Dossier &rarr;</a>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div style="color: var(--alert-red); padding: 20px;">Error loading transaction details.</div>`;
  }
}

function closeInspectModal(e) {
  const modal = document.getElementById("inspect-modal");
  if (modal) modal.classList.remove("open");
}

/* ==========================================================================
   UTILITIES
   ========================================================================== */

function formatTime(isoString) {
  if (!isoString) return "--:--:--";
  try {
    const d = new Date(isoString);
    return d.toTimeString().split(" ")[0];
  } catch {
    return isoString;
  }
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
