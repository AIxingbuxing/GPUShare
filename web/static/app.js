/* 去中心化 GPU 算力共享平台 - 前端逻辑 */
"use strict";

const API = (path, opts = {}) => fetch(path, {
  headers: { "Content-Type": "application/json" },
  ...opts,
}).then(r => r.json()).catch(e => ({ ok: false, error: String(e) }));

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 4) => Number(n || 0).toFixed(d);
const fmtTime = (ts) => {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
};
const shortAddr = (a) => a && a.length > 16 ? a.slice(0, 8) + "..." + a.slice(-8) : (a || "-");

let socket = null;
let gpuChart = null;
let gpuHistory = [];

// ---------- Toast ----------
function toast(msg, type = "") {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show " + type;
  setTimeout(() => { t.className = "toast " + type; }, 3000);
}

// ---------- Tab 切换 ----------
document.querySelectorAll(".nav-item").forEach(el => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(e => e.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(e => e.classList.remove("active"));
    el.classList.add("active");
    $("tab-" + el.dataset.tab).classList.add("active");
    // 切到对应 tab 时刷新数据
    const tab = el.dataset.tab;
    if (tab === "contributor") refreshContributor();
    if (tab === "network") refreshPeers();
    if (tab === "billing") refreshBilling();
    if (tab === "trade") refreshOrders();
    if (tab === "tasks") { refreshComputeStats(); refreshScheduleLogs(); }
  });
});

// ---------- 模态框 Tab ----------
document.querySelectorAll(".mtab").forEach(el => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".mtab").forEach(e => e.classList.remove("active"));
    el.classList.add("active");
    document.querySelectorAll(".mform").forEach(e => e.classList.add("hidden"));
    $(el.dataset.mtab + "Form").classList.remove("hidden");
  });
});

// ---------- 身份 ----------
async function checkAuth() {
  const r = await API("/api/identity/status");
  if (r.ok && r.data.logged_in) {
    $("authMask").classList.add("hidden");
    $("userAddr").textContent = shortAddr(r.data.address);
    $("netDot").classList.add("online");
    refreshAll();
  } else {
    $("authMask").classList.remove("hidden");
  }
}

$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await API("/api/identity/login", {
    method: "POST",
    body: JSON.stringify({ email: fd.get("email"), password: fd.get("password") }),
  });
  if (r.ok) { toast("登录成功", "success"); checkAuth(); }
  else toast(r.error || "登录失败", "error");
});

$("registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await API("/api/identity/register", {
    method: "POST",
    body: JSON.stringify({
      email: fd.get("email"),
      nickname: fd.get("nickname"),
      password: fd.get("password"),
    }),
  });
  if (r.ok) {
    const m = r.data.mnemonic;
    $("mnemonicText").textContent = m;
    $("mnemonicBox").classList.remove("hidden");
    toast("注册成功，请保存助记词", "success");
  } else toast(r.error || "注册失败", "error");
});

$("copyMnemonic").addEventListener("click", () => {
  navigator.clipboard.writeText($("mnemonicText").textContent);
  toast("已复制到剪贴板", "success");
});

$("confirmMnemonic").addEventListener("click", () => {
  $("mnemonicBox").classList.add("hidden");
  checkAuth();
});

$("recoverForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await API("/api/identity/recover", {
    method: "POST",
    body: JSON.stringify({
      email: fd.get("email"),
      mnemonic: fd.get("mnemonic"),
      password: fd.get("password"),
    }),
  });
  if (r.ok) { toast("恢复成功", "success"); checkAuth(); }
  else toast(r.error || "恢复失败", "error");
});

// ---------- 刷新全部 ----------
async function refreshAll() {
  await Promise.all([
    refreshBalance(),
    refreshGpu(),
    refreshPeers(),
    refreshBilling(),
  ]);
}

// ---------- 余额 ----------
async function refreshBalance() {
  const r = await API("/api/balance");
  if (r.ok) {
    $("balanceNum").textContent = fmt(r.data.balance, 2);
    $("stakedNum").textContent = fmt(r.data.staked, 2);
  }
}

// ---------- GPU ----------
async function refreshGpu() {
  const r = await API("/api/gpu/info");
  if (!r.ok) return;
  const g = r.data;
  $("gpuName").textContent = g.device_name + (g.is_cuda ? " (CUDA)" : " (CPU 模拟)");
  $("gpuTops").textContent = fmt(g.estimated_tops, 2);
  $("gpuUtil").textContent = g.utilization.toFixed(1) + "%";
  $("gpuTemp").textContent = g.temperature.toFixed(0) + "℃";
  $("gpuVram").textContent = `${g.used_vram_mb}/${g.total_vram_mb} MB`;
  $("gpuPower").textContent = g.power_usage.toFixed(0) + " W";

  // 图表
  gpuHistory.push(g.utilization);
  if (gpuHistory.length > 30) gpuHistory.shift();
  if (gpuChart) {
    gpuChart.data.datasets[0].data = [...gpuHistory];
    gpuChart.data.labels = gpuHistory.map((_, i) => i);
    gpuChart.update("none");
  }
}

function initGpuChart() {
  const ctx = $("gpuChart").getContext("2d");
  gpuChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: Array.from({ length: 30 }, (_, i) => i),
      datasets: [{
        label: "GPU 利用率 %",
        data: [],
        borderColor: "#2563eb",
        backgroundColor: "rgba(37,99,235,0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { stepSize: 25 } },
        x: { display: false },
      },
    },
  });
}

// ---------- 设置 ----------
async function loadSettings() {
  const r = await API("/api/settings");
  if (!r.ok) return;
  const s = r.data;
  $("localEnabled").checked = s.local_gpu_enabled;
  $("localLimit").value = s.local_utilization_limit;
  $("localLimitVal").textContent = s.local_utilization_limit + "%";
  $("sharingEnabled").checked = s.sharing_enabled;
  $("shareLimit").value = s.sharing_utilization_limit;
  $("shareLimitVal").textContent = s.sharing_utilization_limit + "%";
  $("remoteEnabled").checked = s.remote_enabled || false;
  $("modeSelect").value = s.mode;
  // 互斥提示
  updateMutexHint();
}

async function saveSettings() {
  const r = await API("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      local_gpu_enabled: $("localEnabled").checked,
      local_utilization_limit: parseInt($("localLimit").value),
      sharing_utilization_limit: parseInt($("shareLimit").value),
      mode: $("modeSelect").value,
    }),
  });
  if (r.ok) toast("设置已保存", "success");
}

["localEnabled", "localLimit", "sharingEnabled", "shareLimit", "modeSelect"].forEach(id => {
  const el = $(id);
  el.addEventListener("change", saveSettings);
});
$("localLimit").addEventListener("input", (e) => $("localLimitVal").textContent = e.target.value + "%");
$("shareLimit").addEventListener("input", (e) => $("shareLimitVal").textContent = e.target.value + "%");

// 共享开关（与远程互斥）
$("sharingEnabled").addEventListener("change", async (e) => {
  if (e.target.checked) {
    // 互斥：先关闭远程
    if ($("remoteEnabled").checked) {
      await API("/api/remote/disable", { method: "POST", body: "{}" });
      $("remoteEnabled").checked = false;
    }
    const r = await API("/api/sharing/start", { method: "POST", body: "{}" });
    if (r.ok) { toast("算力共享已开启（已自动关闭远程算力）", "success"); }
    else { toast(r.error || "开启失败", "error"); e.target.checked = false; }
  } else {
    const r = await API("/api/sharing/stop", { method: "POST", body: "{}" });
    if (r.ok) toast("算力共享已关闭", "success");
  }
  refreshPeers();
  updateMutexHint();
});

// 远程算力开关（与共享互斥）
$("remoteEnabled").addEventListener("change", async (e) => {
  if (e.target.checked) {
    // 互斥：先关闭共享
    if ($("sharingEnabled").checked) {
      await API("/api/sharing/stop", { method: "POST", body: "{}" });
      $("sharingEnabled").checked = false;
    }
    const r = await API("/api/remote/enable", { method: "POST", body: "{}" });
    if (r.ok) { toast("远程算力已开启（已自动关闭共享）", "success"); }
    else { toast(r.error || "开启失败", "error"); e.target.checked = false; }
  } else {
    const r = await API("/api/remote/disable", { method: "POST", body: "{}" });
    if (r.ok) toast("远程算力已关闭", "success");
  }
  updateMutexHint();
});

function updateMutexHint() {
  const sharing = $("sharingEnabled").checked;
  const remote = $("remoteEnabled").checked;
  const hint = $("mutexHint");
  if (sharing && remote) {
    hint.style.color = "#dc2626";
    hint.textContent = "冲突！共享与远程不应同时开启";
  } else if (sharing) {
    hint.style.color = "#16a34a";
    hint.textContent = "当前模式：贡献端（对外共享算力）";
  } else if (remote) {
    hint.style.color = "#2563eb";
    hint.textContent = "当前模式：需求端（使用远程算力）";
  } else {
    hint.style.color = "#94a3b8";
    hint.textContent = "共享算力与使用远程算力互斥，不可同时开启";
  }
}

// ---------- 基准测试 ----------
$("btnBenchmark").addEventListener("click", async () => {
  toast("基准测试进行中...");
  const r = await API("/api/gpu/benchmark", { method: "POST", body: JSON.stringify({ duration: 5 }) });
  if (r.ok) toast(`基准测试完成: ${r.data.measured_tops.toFixed(4)} TOPS`, "success");
  else toast("基准测试失败", "error");
});

// ---------- 充值 ----------
$("btnDeposit").addEventListener("click", () => $("depositMask").classList.remove("hidden"));
$("btnCancelDeposit").addEventListener("click", () => $("depositMask").classList.add("hidden"));
$("btnConfirmDeposit").addEventListener("click", async () => {
  const amount = parseFloat($("depositAmount").value);
  const channel = $("depositChannel").value;
  const r = await API("/api/wallet/deposit", {
    method: "POST", body: JSON.stringify({ amount, channel }),
  });
  if (r.ok) {
    toast(`充值成功: ${amount} TPT`, "success");
    $("depositMask").classList.add("hidden");
    refreshBalance();
  } else toast(r.error || "充值失败", "error");
});

// ---------- 转账 ----------
$("btnTransfer").addEventListener("click", () => {
  document.querySelector('[data-tab="trade"]').click();
});
$("btnDoTransfer").addEventListener("click", async () => {
  const to = $("transferTo").value.trim();
  const amount = parseFloat($("transferAmount").value);
  if (!to || !amount) { toast("请填写完整", "error"); return; }
  const r = await API("/api/wallet/transfer", {
    method: "POST", body: JSON.stringify({ to_addr: to, amount }),
  });
  if (r.ok) {
    toast(`转账成功: ${amount} TPT (手续费 ${r.data.fee.toFixed(4)})`, "success");
    $("transferTo").value = "";
    $("transferAmount").value = "";
    refreshBalance();
  } else toast(r.error || "转账失败", "error");
});
$("transferAmount").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value) || 0;
  $("transferFee").textContent = `手续费: ${(v * 0.02).toFixed(4)} TPT`;
});

// ---------- 质押 ----------
$("btnStake").addEventListener("click", async () => {
  const amount = prompt("质押金额 (TPT):", "100");
  if (!amount) return;
  const r = await API("/api/wallet/stake", {
    method: "POST", body: JSON.stringify({ amount: parseFloat(amount) }),
  });
  if (r.ok) { toast(`质押成功: ${amount} TPT`, "success"); refreshBalance(); }
  else toast(r.error || "质押失败", "error");
});

// ---------- 节点 ----------
async function refreshPeers() {
  const r = await API("/api/peers");
  if (!r.ok) return;
  const peers = r.data;
  $("peerCount").textContent = peers.length;
  const sharing = peers.filter(p => p.is_sharing);
  $("sharingCount").textContent = sharing.length;
  const totalTops = peers.reduce((s, p) => s + (p.gpu_tops || 0), 0);
  $("totalTops").textContent = totalTops.toFixed(1);
  const avgLatency = peers.length ?
    (peers.reduce((s, p) => s + p.latency_ms, 0) / peers.length).toFixed(0) : 0;
  $("avgLatency").textContent = avgLatency;

  const tbody = $("peersTbody");
  tbody.innerHTML = "";
  if (peers.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:24px;">暂无其他节点，请启动第二个节点连接</td></tr>';
    return;
  }
  for (const p of peers) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${shortAddr(p.address)}</td>
      <td>${p.nickname || "-"}</td>
      <td>${p.gpu_name || "-"}</td>
      <td>${(p.gpu_tops || 0).toFixed(2)}</td>
      <td>${(p.latency_ms || 0).toFixed(0)} ms</td>
      <td>${p.is_sharing ? '<span class="tag green">是</span>' : '<span class="tag gray">否</span>'}</td>
      <td>${p.reputation || 100}</td>
      <td>${p.is_online ? '<span class="tag green">在线</span>' : '<span class="tag red">离线</span>'}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---------- 账单 ----------
async function refreshBilling() {
  const [txR, stR] = await Promise.all([
    API("/api/transactions?limit=50"),
    API("/api/settlements"),
  ]);
  const txs = txR.ok ? txR.data : [];
  const settlements = stR.ok ? stR.data : [];

  $("txCount").textContent = txs.length;
  $("settlementCount").textContent = settlements.length;
  let earned = 0, consumed = 0;
  for (const t of txs) {
    if (t.type === "reward" || t.type === "deposit" || t.type === "new_user_bonus") earned += t.amount;
    if (t.type === "consume" || t.type === "transfer") consumed += t.amount + (t.fee || 0);
  }
  $("totalEarned").textContent = earned.toFixed(2);
  $("totalConsumed").textContent = consumed.toFixed(2);

  const tbody = $("txTbody");
  tbody.innerHTML = "";
  if (txs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:24px;">暂无交易记录</td></tr>';
  } else {
    for (const t of txs) {
      const direction = t.from_addr === (window._myAddr || "") ? "支出" : "收入";
      const typeMap = {
        deposit: "充值", consume: "消费", reward: "收益", transfer: "转账",
        stake: "质押", unstake: "解押", slash: "扣除", fee: "手续费",
        trade: "交易", new_user_bonus: "新用户福利",
      };
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${fmtTime(t.timestamp)}</td>
        <td><span class="tag blue">${typeMap[t.type] || t.type}</span></td>
        <td>${direction}</td>
        <td class="mono">${shortAddr(t.from_addr === (window._myAddr || "") ? t.to_addr : t.from_addr)}</td>
        <td>${t.amount.toFixed(4)}</td>
        <td>${(t.fee || 0).toFixed(4)}</td>
        <td><span class="tag ${t.status === 'confirmed' ? 'green' : 'yellow'}">${t.status}</span></td>
      `;
      tbody.appendChild(tr);
    }
  }

  const stBody = $("settlementTbody");
  stBody.innerHTML = "";
  if (settlements.length === 0) {
    stBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:24px;">暂无结算单</td></tr>';
  } else {
    for (const s of settlements) {
      const statusMap = {
        pending: ["待结算", "yellow"], challenge: ["挑战期", "yellow"],
        confirmed: ["已确认", "green"], slashed: ["已扣除", "red"],
        disputed: ["争议中", "red"], cancelled: ["已取消", "gray"],
      };
      const [statusText, statusColor] = statusMap[s.status] || [s.status, "gray"];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">${shortAddr(s.settlement_id)}</td>
        <td class="mono">${shortAddr(s.task_id)}</td>
        <td class="mono">${shortAddr(s.contributor)}</td>
        <td>${(s.measured_tops || 0).toFixed(4)}</td>
        <td>${(s.duration_min || 0).toFixed(2)}</td>
        <td>${(s.total_tpt || 0).toFixed(4)}</td>
        <td>${(s.contributor_reward || 0).toFixed(4)}</td>
        <td><span class="tag ${statusColor}">${statusText}</span></td>
      `;
      stBody.appendChild(tr);
    }
  }
}

// ---------- 订单 ----------
async function refreshOrders() {
  const r = await API("/api/orders");
  if (!r.ok) return;
  const tbody = $("ordersTbody");
  tbody.innerHTML = "";
  if (r.data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:24px;">暂无挂单</td></tr>';
    return;
  }
  for (const o of r.data) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="tag ${o.type === 'buy' ? 'green' : 'red'}">${o.type === 'buy' ? '买入' : '卖出'}</span></td>
      <td class="mono">${shortAddr(o.maker)}</td>
      <td>¥${o.price.toFixed(2)}</td>
      <td>${o.amount.toFixed(2)}</td>
      <td>${o.filled.toFixed(2)}</td>
      <td><span class="tag ${o.status === 'open' ? 'blue' : 'gray'}">${o.status}</span></td>
      <td><button class="btn small" data-id="${o.order_id}">撤销</button></td>
    `;
    tr.querySelector("button").addEventListener("click", async () => {
      const r2 = await API("/api/orders/cancel", { method: "POST", body: JSON.stringify({ order_id: o.order_id }) });
      if (r2.ok) { toast("已撤销", "success"); refreshOrders(); }
      else toast("撤销失败", "error");
    });
    tbody.appendChild(tr);
  }
}

$("btnPlaceOrder").addEventListener("click", async () => {
  const r = await API("/api/orders/place", {
    method: "POST",
    body: JSON.stringify({
      type: $("orderType").value,
      price: parseFloat($("orderPrice").value),
      amount: parseFloat($("orderAmount").value),
    }),
  });
  if (r.ok) { toast("挂单成功", "success"); refreshOrders(); }
  else toast(r.error || "挂单失败", "error");
});

// ---------- 贡献端 ----------
async function refreshContributor() {
  const [statsR, histR, gpuR, balR] = await Promise.all([
    API("/api/contributor/stats"),
    API("/api/contributor/history?limit=30"),
    API("/api/gpu/info"),
    API("/api/balance"),
  ]);

  // 统计
  if (statsR.ok) {
    const s = statsR.data;
    $("contribTotalEarned").textContent = (s.confirmed_reward || 0).toFixed(4);
    $("contribPending").textContent = (s.pending_reward || 0).toFixed(4);
    $("contribTasks").textContent = s.tasks_completed || 0;
    $("contribSuccess").textContent = (s.success_rate || 100).toFixed(0) + "%";
    $("contribSharingEnabled").checked = s.sharing_enabled;
    $("contribGpuLimit").value = s.sharing_utilization_limit || 80;
    $("contribGpuLimitVal").textContent = (s.sharing_utilization_limit || 80) + "%";
    $("contribActive").textContent = s.active_tasks || 0;
    const dur = s.sharing_duration || 0;
    $("contribDuration").textContent = dur > 60 ? (dur/60).toFixed(0)+"m" : dur.toFixed(0)+"s";
  }

  // 余额（质押）
  if (balR.ok) {
    $("contribStaked").textContent = (balR.data.staked || 0).toFixed(2) + " TPT";
  }

  // GPU 信息
  if (gpuR.ok) {
    const g = gpuR.data;
    $("contribGpuName").textContent = g.device_name + (g.is_cuda ? " (CUDA)" : " (CPU)");
    $("contribGpuUtil").textContent = g.utilization.toFixed(1) + "%";
    $("contribGpuTemp").textContent = g.temperature.toFixed(0) + "℃";
    $("contribGpuVram").textContent = g.used_vram_mb + "/" + g.total_vram_mb + " MB";
    $("contribGpuBar").style.width = g.utilization + "%";
  }

  // 历史记录
  const tbody = $("contribHistoryTbody");
  tbody.innerHTML = "";
  if (!histR.ok || histR.data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:24px;">暂无贡献记录，开启共享后自动接取远程任务</td></tr>';
    return;
  }
  for (const h of histR.data) {
    const statusMap = {
      completed: ["完成", "green"], failed: ["失败", "red"],
      running: ["执行中", "blue"], timeout: ["超时", "yellow"],
    };
    const [st, c] = statusMap[h.status] || [h.status, "gray"];
    // 估算收益：实测TOPS × 1.0 × 时长(分钟) × 98%
    const tpt = h.tops_measured && h.duration_sec
      ? (h.tops_measured * (h.duration_sec / 60) * 0.98).toFixed(6)
      : "0.000000";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmtTime(h.timestamp)}</td>
      <td>${h.task_type || "-"}</td>
      <td class="mono">${shortAddr(h.requester)}</td>
      <td><span class="tag ${c}">${st}</span></td>
      <td>${(h.duration_sec || 0).toFixed(2)}s</td>
      <td>${(h.tops_measured || 0).toFixed(4)}</td>
      <td style="color:#16a34a;font-weight:600;">${tpt}</td>
    `;
    tbody.appendChild(tr);
  }
}

// 贡献端共享开关（与远程互斥）
$("contribSharingEnabled").addEventListener("change", async (e) => {
  if (e.target.checked) {
    // 互斥：先关闭远程
    if ($("remoteEnabled") && $("remoteEnabled").checked) {
      await API("/api/remote/disable", { method: "POST", body: "{}" });
      $("remoteEnabled").checked = false;
    }
    const r = await API("/api/sharing/start", { method: "POST", body: "{}" });
    if (r.ok) { toast("算力共享已开启（无需质押）", "success"); refreshContributor(); }
    else { toast(r.error || "开启失败", "error"); e.target.checked = false; }
  } else {
    const r = await API("/api/sharing/stop", { method: "POST", body: "{}" });
    if (r.ok) { toast("算力共享已关闭", "success"); refreshContributor(); }
  }
  // 同步总览页开关
  if ($("sharingEnabled")) $("sharingEnabled").checked = e.target.checked;
  updateMutexHint();
});

// 贡献端 GPU 使用率
$("contribGpuLimit").addEventListener("input", (e) => {
  $("contribGpuLimitVal").textContent = e.target.value + "%";
});
$("contribGpuLimit").addEventListener("change", async () => {
  await API("/api/settings", {
    method: "POST",
    body: JSON.stringify({ sharing_utilization_limit: parseInt($("contribGpuLimit").value) }),
  });
  toast("GPU 使用率上限已更新", "success");
});

// 追加质押
$("btnContribStake").addEventListener("click", async () => {
  const amount = prompt("追加质押金额 (TPT):", "100");
  if (!amount) return;
  const r = await API("/api/wallet/stake", {
    method: "POST", body: JSON.stringify({ amount: parseFloat(amount) }),
  });
  if (r.ok) { toast(`质押成功: ${amount} TPT`, "success"); refreshContributor(); }
  else toast(r.error || "质押失败", "error");
});

// 解除质押
$("btnContribUnstake").addEventListener("click", async () => {
  const amount = prompt("解除质押金额 (TPT):", "100");
  if (!amount) return;
  const r = await API("/api/wallet/unstake", {
    method: "POST", body: JSON.stringify({ amount: parseFloat(amount) }),
  });
  if (r.ok) { toast(`解除质押成功: ${amount} TPT`, "success"); refreshContributor(); }
  else toast(r.error || "解除质押失败", "error");
});

// ---------- 计算调度监控 ----------
let computeGpuChart = null;
let computeGpuHistory = [];

async function refreshComputeStats() {
  const r = await API("/api/compute/stats");
  if (!r.ok) return;
  const s = r.data;
  $("computeTotal").textContent = s.total_requests || 0;
  $("computeLocal").textContent = s.local_executed || 0;
  $("computeRemote").textContent = (s.remote_executed || 0) + (s.mixed_executed || 0);
  $("computeSuccess").textContent = ((s.success_rate || 1) * 100).toFixed(0) + "%";

  // GPU 利用率
  const util = s.current_gpu_util || 0;
  $("currentGpuUtil").textContent = util.toFixed(1) + "%";
  $("gpuProgressBar").style.width = util + "%";

  // GPU 趋势图
  if (s.gpu_trend && s.gpu_trend.length > 0) {
    computeGpuHistory = s.gpu_trend;
    if (computeGpuChart) {
      computeGpuChart.data.datasets[0].data = [...computeGpuHistory];
      computeGpuChart.data.labels = computeGpuHistory.map((_, i) => i);
      computeGpuChart.update("none");
    }
  }
}

async function refreshScheduleLogs() {
  const r = await API("/api/compute/logs?limit=30");
  if (!r.ok) return;
  const tbody = $("scheduleLogsTbody");
  tbody.innerHTML = "";
  if (r.data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#94a3b8;padding:24px;">暂无调度记录，外部软件通过 API/SDK 发起计算请求后将显示</td></tr>';
    return;
  }
  for (const log of r.data) {
    const scheduleColor = {
      local: "green", remote: "yellow", mixed: "blue",
      local_fallback: "yellow",
    };
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmtTime(log.timestamp)}</td>
      <td>${log.source || "-"}</td>
      <td><span class="tag blue">${log.type || "-"}</span></td>
      <td><span class="tag ${scheduleColor[log.schedule] || "gray"}">${log.schedule}</span></td>
      <td>${(log.gpu_util || 0).toFixed(1)}%</td>
      <td style="font-size:12px;color:#64748b;">${log.reason || "-"}</td>
    `;
    tbody.appendChild(tr);
  }
}

function initComputeGpuChart() {
  const ctx = document.getElementById("computeGpuChart");
  if (!ctx) return;
  computeGpuChart = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: {
      labels: Array.from({ length: 60 }, (_, i) => i),
      datasets: [{
        label: "GPU 利用率 %",
        data: [],
        borderColor: "#2563eb",
        backgroundColor: "rgba(37,99,235,0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      }, {
        label: "远程阈值 75%",
        data: Array(60).fill(75),
        borderColor: "rgba(220,38,38,0.3)",
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { stepSize: 25 } },
        x: { display: false },
      },
    },
  });
}

// ---------- WebSocket ----------
function initSocket() {
  socket = io();
  socket.on("status", (data) => {
    if (data.balance) {
      $("balanceNum").textContent = fmt(data.balance.balance, 2);
      $("stakedNum").textContent = fmt(data.balance.staked, 2);
    }
    if (data.gpu) {
      const g = data.gpu;
      $("gpuName").textContent = g.device_name + (g.is_cuda ? " (CUDA)" : " (CPU 模拟)");
      $("gpuTops").textContent = fmt(g.estimated_tops, 2);
      $("gpuUtil").textContent = g.utilization.toFixed(1) + "%";
      $("gpuTemp").textContent = g.temperature.toFixed(0) + "℃";
      $("gpuVram").textContent = `${g.used_vram_mb}/${g.total_vram_mb} MB`;
      $("gpuPower").textContent = g.power_usage.toFixed(0) + " W";
      gpuHistory.push(g.utilization);
      if (gpuHistory.length > 30) gpuHistory.shift();
      if (gpuChart) {
        gpuChart.data.datasets[0].data = [...gpuHistory];
        gpuChart.data.labels = gpuHistory.map((_, i) => i);
        gpuChart.update("none");
      }
    }
    if (data.stats) {
      $("peerCount").textContent = data.stats.peer_count || 0;
      $("sharingCount").textContent = data.stats.sharing_peer_count || 0;
    }
  });
}

// ---------- 初始化 ----------
async function init() {
  initGpuChart();
  initComputeGpuChart();
  initSocket();
  await checkAuth();
  await loadSettings();
  // 周期刷新
  setInterval(refreshPeers, 10000);
  setInterval(refreshBilling, 15000);
  setInterval(refreshContributor, 5000);   // 贡献端 5 秒刷新
  setInterval(refreshComputeStats, 3000);   // 计算调度 3 秒刷新
  setInterval(refreshScheduleLogs, 5000);   // 调度日志 5 秒刷新
  // 获取自己地址
  const r = await API("/api/identity/status");
  if (r.ok && r.data.address) window._myAddr = r.data.address;
}

init();
