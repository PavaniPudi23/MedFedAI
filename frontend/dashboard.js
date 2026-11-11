const $ = (selector) => document.querySelector(selector);
const apiStatus = $('#api-status');
const startButton = $('#start-run');
const runLabel = $('#run-button-label');
const roundInput = $('#rounds-input');
const noiseInput = $('#noise-input');
const history = [];
const previewClients = [
  {id: 'HOSP-01', name: 'Northside Hospital', type: 'Hospital', patients: 1240, color: 'mint'},
  {id: 'CLIN-02', name: 'Riverbend Clinic', type: 'Clinic', patients: 860, color: 'blue'},
  {id: 'CENT-03', name: 'Central Imaging Center', type: 'Imaging center', patients: 1530, color: 'gold'},
];

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderClients(clients) {
  $('#client-total').textContent = String(clients.length).padStart(2, '0');
  $('#client-list').innerHTML = clients.map((client) => `
    <div class="client-row">
      <span class="client-logo ${client.color}">${client.type === 'Hospital' ? '✚' : client.type === 'Clinic' ? '⌂' : '✣'}</span>
      <span><span class="client-name">${client.name}</span><small class="client-sub">${client.id} · ${client.patients.toLocaleString()} sample records</small></span>
      <span class="client-state" data-client-state="${client.id}">Ready</span>
      <span class="client-sent">Images shared: 0</span>
    </div>`).join('');
}

async function loadOverview() {
  try {
    const response = await fetch('/api/overview');
    if (!response.ok) throw new Error('API unavailable');
    const overview = await response.json();
    renderClients(overview.clients);
    apiStatus.textContent = 'Connected · simulation';
  } catch (error) {
    apiStatus.textContent = 'Preview mode · API offline';
    renderClients(previewClients);
  }
}

function drawChart() {
  const canvas = $('#training-chart');
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * pixelRatio);
  canvas.height = Math.floor(rect.height * pixelRatio);
  const ctx = canvas.getContext('2d');
  ctx.scale(pixelRatio, pixelRatio);
  const width = rect.width;
  const height = rect.height;
  const left = 27, right = 10, top = 12, bottom = 20;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.font = '8px DM Sans, Arial';
  ctx.fillStyle = '#a5b0ae';
  ctx.strokeStyle = '#edf1ef';
  ctx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const y = top + plotHeight * fraction;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    ctx.fillText(`${(1 - fraction).toFixed(2)}`, 1, y + 3);
  });
  if (history.length < 1) return;
  const xAt = (index) => left + (history.length === 1 ? plotWidth / 2 : index * plotWidth / (history.length - 1));
  const yAt = (value) => top + (1 - Math.max(0, Math.min(1, value))) * plotHeight;
  const line = (values, color, key) => {
    ctx.beginPath();
    values.forEach((point, index) => index ? ctx.lineTo(xAt(index), yAt(point[key])) : ctx.moveTo(xAt(index), yAt(point[key])));
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    values.forEach((point, index) => {
      ctx.beginPath(); ctx.arc(xAt(index), yAt(point[key]), 3, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  };
  line(history.map((item) => ({value: item.validation_accuracy})), '#258a78', 'value');
  line(history.map((item) => ({value: item.train_loss})), '#dfa75d', 'value');
}

function updateRound(record) {
  history.push({validation_accuracy: record.validation_accuracy, train_loss: record.train_loss});
  $('#chart-empty').hidden = true;
  $('#round-total').textContent = `${record.round}`;
  $('#round-sub').textContent = `of ${roundInput.value} demo rounds`;
  $('#accuracy-total').innerHTML = `${Math.round(record.validation_accuracy * 100)}<small>%</small>`;
  $('#chart-round-label').textContent = `ROUND ${String(record.round).padStart(2, '0')} · ${record.progress}% COMPLETE`;
  $('#run-state').textContent = 'RUNNING';
  document.querySelectorAll('[data-client-state]').forEach((element) => {
    const active = record.sampled_clients.includes(element.dataset.clientState);
    element.textContent = active ? 'Update shared' : 'Waiting';
    element.style.color = active ? '#4d977e' : '#9ba7a4';
    element.style.background = active ? '#edf6f1' : '#f4f6f5';
  });
  const row = document.createElement('div');
  row.className = 'log-item';
  row.innerHTML = `<b>Round ${String(record.round).padStart(2, '0')} <span>${record.client_count} clients</span></b><p>Illustrative accuracy ${percent(record.validation_accuracy)} · raw images transferred ${record.raw_images_transferred}.</p>`;
  $('#log-list').prepend(row);
  drawChart();
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runDemo() {
  startButton.disabled = true;
  runLabel.textContent = 'Starting demo…';
  history.length = 0;
  $('#log-list').innerHTML = '';
  $('#round-total').textContent = '0';
  $('#accuracy-total').innerHTML = '—<small>%</small>';
  $('#chart-empty').hidden = false;
  $('#chart-round-label').textContent = 'PREPARING DEMO';
  try {
    const response = await fetch('/api/demo/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        rounds: Number(roundInput.value),
        sampling_rate: 0.67,
        clip_norm: 1.0,
        noise_multiplier: Number(noiseInput.value),
        local_epochs: 1,
      }),
    });
    if (!response.ok) throw new Error('Could not start demo run. Is the backend running?');
    const {run_id: runId} = await response.json();
    runLabel.textContent = 'Demo running…';
    $('#run-state').textContent = 'RUNNING';
    let lastRound = 0;
    while (true) {
      await delay(400);
      const runResponse = await fetch(`/api/demo/run/${runId}`);
      if (!runResponse.ok) throw new Error('The demo run stopped unexpectedly.');
      const run = await runResponse.json();
      for (const item of run.history) {
        if (item.round > lastRound) {
          updateRound(item);
          lastRound = item.round;
        }
      }
      if (run.status === 'completed') break;
    }
    runLabel.textContent = 'Run again';
    $('#run-state').textContent = 'COMPLETED';
    $('#chart-round-label').textContent = `${history.length} SIMULATED ROUNDS · COMPLETE`;
    document.querySelectorAll('[data-client-state]').forEach((element) => {
      element.textContent = 'Ready';
      element.style.color = '';
      element.style.background = '';
    });
  } catch (error) {
    // A file:// preview cannot call the local API. Keep the UI demo usable and
    // clearly mark the run as browser-side simulation.
    apiStatus.textContent = 'Preview mode · API offline';
    runLabel.textContent = 'Preview running…';
    $('#run-state').textContent = 'LOCAL PREVIEW';
    const rounds = Number(roundInput.value);
    const rate = 0.67;
    for (let index = 1; index <= rounds; index += 1) {
      await delay(420);
      const selected = previewClients.filter((client, position) => ((index + position) % 3) !== 0 || rate >= 1);
      const accuracy = Math.min(0.89, 0.56 + 0.30 * (1 - (0.79 ** index)) - Math.max(0, Number(noiseInput.value) - 1) * 0.025);
      const record = {
        round: index,
        progress: Math.round(index / rounds * 100),
        sampled_clients: (selected.length ? selected : [previewClients[index % previewClients.length]]).map((client) => client.id),
        client_count: selected.length || 1,
        train_loss: Math.max(0.17, 0.95 * (0.79 ** index)),
        validation_accuracy: accuracy,
        raw_images_transferred: 0,
        protection: Number(noiseInput.value) > 0 ? 'illustrative clipping + noise settings' : 'privacy disabled',
      };
      updateRound(record);
    }
    runLabel.textContent = 'Run again';
    $('#run-state').textContent = 'LOCAL PREVIEW';
    $('#chart-round-label').textContent = `${rounds} BROWSER-SIMULATED ROUNDS`;
    document.querySelectorAll('[data-client-state]').forEach((element) => {
      element.textContent = 'Ready';
      element.style.color = '';
      element.style.background = '';
    });
  } finally {
    startButton.disabled = false;
  }
}

roundInput.addEventListener('input', () => { $('#rounds-value').textContent = roundInput.value; });
noiseInput.addEventListener('input', () => { $('#noise-value').textContent = Number(noiseInput.value).toFixed(1); });
startButton.addEventListener('click', runDemo);
$('#privacy-info').addEventListener('click', () => $('#privacy-dialog').showModal());
$('#dialog-close').addEventListener('click', () => $('#privacy-dialog').close());
$('#dialog-ok').addEventListener('click', () => $('#privacy-dialog').close());
window.addEventListener('resize', drawChart);
loadOverview();
