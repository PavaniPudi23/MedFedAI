const $ = (selector) => document.querySelector(selector);
const apiStatus = $('#api-status');
const runLabel = $('#run-button-label');
const roundInput = $('#rounds-input');
const noiseInput = $('#noise-input');
const clipInput = $('#clip-input');
const deltaInput = $('#delta-input');
const epochsInput = $('#epochs-input');
const datasetInput = $('#dataset-select');
const folderInput = $('#dataset-folder');
const importButton = $('#import-dataset');
const phaseIndex = $('#phase-index');
const phaseButton = $('#show-phases');
const phaseStatus = $('#phase-status');
const history = [];
let activeDatasetId = '';

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatEpsilon(value) {
  if (value === 'infinite') return 'infinite';
  const epsilon = Number(value);
  return Number.isFinite(epsilon) ? epsilon.toFixed(2) : String(value);
}

function setChartEmpty(visible) {
  const emptyState = $('#chart-empty');
  emptyState.hidden = !visible;
  emptyState.style.display = visible ? 'flex' : 'none';
}

function renderClients(clients) {
  $('#client-total').textContent = String(clients.length).padStart(2, '0');
  const clientCount = document.querySelector('.client-count');
  if (clientCount) clientCount.textContent = `${String(clients.length).padStart(2, '0')} CLIENTS`;
  $('#client-list').innerHTML = clients.map((client) => `
    <div class="client-row">
      <span class="client-logo ${client.color}">⌂</span>
      <span><span class="client-name">${client.name}</span><small class="client-sub">${client.id} · Local Deep CapsNet · ${Number(client.image_count).toLocaleString()} local images</small></span>
      <span class="client-state" data-client-state="${client.id}">Ready</span>
      <span class="client-sent">Model update only</span>
    </div>`).join('');
}

const pages = ['overview', 'federation', 'model', 'privacy', 'activity'];
const pageTitles = {overview: 'Overview', federation: 'Federation', model: 'Model architecture', privacy: 'Privacy settings', activity: 'Experiment log'};
function showPage() {
  const requested = window.location.hash.slice(1);
  const page = pages.includes(requested) ? requested : 'overview';
  document.querySelectorAll('[data-page]').forEach((element) => {
    const active = element.dataset.page === page;
    element.classList.toggle('active', active);
    element.hidden = !active;
  });
  document.querySelectorAll('[data-nav]').forEach((element) => {
    const active = element.dataset.nav === page;
    element.classList.toggle('active', active);
    if (active) element.setAttribute('aria-current', 'page');
    else element.removeAttribute('aria-current');
  });
  $('#page-breadcrumb').textContent = pageTitles[page];
  if (page === 'overview') requestAnimationFrame(drawChart);
}

async function loadOverview(preferredDatasetId = '') {
  try {
    const requestedId = preferredDatasetId || datasetInput.value || localStorage.getItem('medfedai-dataset') || '';
    const query = requestedId ? `?dataset_id=${encodeURIComponent(requestedId)}` : '';
    const response = await fetch(`/api/overview${query}`);
    if (!response.ok) throw new Error('API unavailable');
    const overview = await response.json();
    const previousId = activeDatasetId;
    datasetInput.replaceChildren(...(overview.datasets || []).map((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.id;
      option.textContent = dataset.label;
      return option;
    }));
    datasetInput.value = overview.dataset_id || '';
    activeDatasetId = overview.dataset_id || '';
    datasetInput.disabled = (overview.datasets || []).length < 2;
    if (overview.dataset_id) localStorage.setItem('medfedai-dataset', overview.dataset_id);
    const selectedLabel = overview.datasets?.find((dataset) => dataset.id === overview.dataset_id)?.label || overview.dataset;
    $('#dataset-description').textContent = overview.model_ready
      ? `${selectedLabel} · ${overview.clients.reduce((sum, client) => sum + Number(client.image_count || 0), 0).toLocaleString()} client images · ${overview.test_image_count.toLocaleString()} held-out test images`
      : `${selectedLabel} is not prepared for training.`;
    renderClients(overview.clients);
    apiStatus.textContent = 'Connected · local training';
    $('#data-status').textContent = overview.model_ready ? 'LOCAL DATA READY' : 'DATA NOT READY';
    document.querySelectorAll('[data-start-run]').forEach((button) => { button.disabled = !overview.model_ready; });
    phaseIndex.max = String(Math.max(0, overview.test_image_count - 1));
    phaseButton.disabled = !overview.checkpoint_ready;
    phaseStatus.textContent = overview.checkpoint_ready
      ? `${overview.test_image_count.toLocaleString()} held-out images available locally.`
      : 'Train the model to create a checkpoint first.';
    if (!overview.model_ready) $('#run-state').textContent = 'ADD DATA';
    if (previousId && previousId !== overview.dataset_id) {
      history.length = 0;
      setChartEmpty(true);
      $('#log-list').innerHTML = '<div class="log-empty">Dataset changed. Start a run to see its measured results.</div>';
      $('#round-total').textContent = '—';
      $('#accuracy-total').innerHTML = '—<small>%</small>';
      $('#round-sub').textContent = 'Start a local training run';
      $('#accuracy-sub').textContent = 'Measured on held-out images';
      $('#chart-round-label').textContent = 'WAITING FOR RUN';
      $('#phase-result').hidden = true;
      phaseButton.disabled = !overview.checkpoint_ready;
    }
  } catch (error) {
    apiStatus.textContent = 'Backend offline · open the local server';
    $('#data-status').textContent = 'BACKEND OFFLINE';
    document.querySelectorAll('[data-start-run]').forEach((button) => { button.disabled = true; });
  }
}

function describeSelectedFolder() {
  const images = [...(folderInput.files || [])].filter((file) => /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(file.name));
  const totalSize = images.reduce((sum, file) => sum + file.size, 0);
  const sizeLabel = totalSize >= 1024 ** 3 ? `${(totalSize / 1024 ** 3).toFixed(2)} GB` : `${(totalSize / 1024 ** 2).toFixed(1)} MB`;
  importButton.disabled = images.length === 0;
  $('#import-status').textContent = images.length
    ? `${images.length.toLocaleString()} image files selected · ${sizeLabel}. Files are sent only to this computer’s local app.`
    : 'Choose a folder containing labeled image files.';
}

function importSelectedDataset() {
  const selected = [...(folderInput.files || [])].filter((file) => /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(file.name));
  if (!selected.length) return;
  const manifest = JSON.stringify({files: selected.map((file) => ({path: file.webkitRelativePath || file.name, size: file.size}))});
  const payloadParts = [`${manifest}\n`, ...selected];
  const payload = new Blob(payloadParts, {type: 'application/x-medfedai-folder'});
  const datasetName = $('#dataset-name').value.trim() || 'My medical image dataset';
  importButton.disabled = true;
  folderInput.disabled = true;
  $('#import-progress').hidden = false;
  $('#import-progress').value = 0;
  $('#import-status').textContent = 'Uploading files to the local app and arranging client folders…';
  const request = new XMLHttpRequest();
  request.open('POST', `/api/datasets/import?dataset_name=${encodeURIComponent(datasetName)}&client_count=${encodeURIComponent($('#client-count').value)}`);
  request.setRequestHeader('Content-Type', 'application/x-medfedai-folder');
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      const percentDone = Math.round(event.loaded * 100 / event.total);
      $('#import-progress').value = percentDone;
      $('#import-status').textContent = `Sending files to the local app… ${percentDone}%`;
    }
  };
  request.onload = () => {
    let result = {};
    try { result = JSON.parse(request.responseText); } catch {}
    if (request.status < 200 || request.status >= 300) {
      $('#import-status').textContent = result.detail || 'Import failed. Check that the selected folder contains class-labeled images.';
      importButton.disabled = false;
      folderInput.disabled = false;
      return;
    }
    localStorage.setItem('medfedai-dataset', result.dataset_id);
    $('#import-status').textContent = `Imported “${result.name}”: ${result.client_count} clients and ${Number(result.test_image_count).toLocaleString()} held-out test images. Dataset is stored locally.`;
    folderInput.value = '';
    loadOverview(result.dataset_id);
  };
  request.onerror = () => {
    $('#import-status').textContent = 'Could not connect to the local app. Make sure the backend is running.';
    importButton.disabled = false;
    folderInput.disabled = false;
  };
  request.onloadend = () => {
    $('#import-progress').hidden = true;
    folderInput.disabled = false;
  };
  request.send(payload);
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
  const lossScale = Math.max(1, ...history.map((item) => item.train_loss));
  const yAt = (value, scaleMax) => top + (1 - Math.max(0, Math.min(1, value / scaleMax))) * plotHeight;
  const line = (values, color, scaleMax) => {
    ctx.beginPath();
    values.forEach((point, index) => index ? ctx.lineTo(xAt(index), yAt(point.value, scaleMax)) : ctx.moveTo(xAt(index), yAt(point.value, scaleMax)));
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    values.forEach((point, index) => {
      ctx.beginPath(); ctx.arc(xAt(index), yAt(point.value, scaleMax), 3, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  };
  line(history.map((item) => ({value: item.validation_accuracy})), '#258a78', 1);
  line(history.map((item) => ({value: item.train_loss})), '#dfa75d', lossScale);
}

function updateRound(record) {
  history.push({
    round: record.round,
    client_count: record.client_count,
    test_accuracy: record.validation_accuracy,
    macro_f1: record.macro_f1,
    test_loss: record.train_loss,
    epsilon: record.epsilon,
    delta: record.delta,
    validation_accuracy: record.validation_accuracy,
    train_loss: record.train_loss,
  });
  setChartEmpty(false);
  $('#round-total').textContent = `${record.round}`;
  $('#round-sub').textContent = `of ${roundInput.value} training rounds`;
  $('#accuracy-total').innerHTML = `${Math.round(record.validation_accuracy * 100)}<small>%</small>`;
  $('#accuracy-sub').textContent = `Macro F1 ${percent(record.macro_f1)} · ε ${formatEpsilon(record.epsilon)}`;
  $('#chart-round-label').textContent = `ROUND ${String(record.round).padStart(2, '0')} · ${record.progress}% COMPLETE`;
  $('#run-state').textContent = 'RUNNING';
  $('#global-model-status').textContent = `Global model updated · round ${record.round}`;
  document.querySelectorAll('[data-client-state]').forEach((element) => {
    const active = record.sampled_clients.includes(element.dataset.clientState);
    element.textContent = active ? 'Update shared' : 'Waiting';
    element.style.color = active ? '#4d977e' : '#9ba7a4';
    element.style.background = active ? '#edf6f1' : '#f4f6f5';
  });
  const row = document.createElement('div');
  row.className = 'log-item';
  const epsilonText = record.epsilon === 'infinite' ? 'ε infinite (no DP)' : `ε ${formatEpsilon(record.epsilon)}`;
  row.innerHTML = `<b>Round ${String(record.round).padStart(2, '0')} <span>${record.client_count} clients</span></b><p>Measured test accuracy ${percent(record.validation_accuracy)} · macro F1 ${percent(record.macro_f1)} · ${epsilonText} · raw images transferred 0.</p>`;
  $('#log-list').prepend(row);
  drawChart();
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runModel() {
  document.querySelectorAll('[data-start-run]').forEach((button) => { button.disabled = true; });
  datasetInput.disabled = true;
  runLabel.textContent = 'Starting model…';
  history.length = 0;
  $('#log-list').innerHTML = '';
  $('#round-total').textContent = '0';
  $('#accuracy-total').innerHTML = '—<small>%</small>';
  $('#accuracy-sub').textContent = 'Measured on held-out images';
  setChartEmpty(true);
  $('#chart-round-label').textContent = 'PREPARING MODEL';
  try {
    const response = await fetch('/api/model/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset_id: datasetInput.value,
        rounds: Number(roundInput.value),
        local_epochs: Number(epochsInput.value),
        clip_norm: Number(clipInput.value),
        noise_multiplier: Number(noiseInput.value),
        delta: Number(deltaInput.value),
      }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || 'Could not start model training. Check data folders and dependencies.');
    }
    const {run_id: runId} = await response.json();
    runLabel.textContent = 'Training…';
    $('#run-state').textContent = 'RUNNING';
    let lastRound = 0;
    while (true) {
      await delay(800);
      const runResponse = await fetch(`/api/model/run/${runId}`);
      if (!runResponse.ok) throw new Error('The model run stopped unexpectedly.');
      const run = await runResponse.json();
      for (const item of run.history) {
        if (item.round > lastRound) {
          updateRound(item);
          lastRound = item.round;
        }
      }
      if (run.status === 'failed') throw new Error(run.error || 'Model training failed. Check the local server log.');
      if (run.status === 'completed') {
        $('#chart-round-label').textContent = `${history.length} MEASURED ROUNDS · COMPLETE`;
        $('#run-state').textContent = 'COMPLETED';
        $('#global-model-status').textContent = `Updated global model · round ${history.length} complete`;
        runLabel.textContent = 'Run again';
        const output = document.createElement('div');
        output.className = 'log-item';
        output.innerHTML = `<b>Saved experiment outputs</b><p>${run.output_dir}\metrics.csv · privacy_report.json · model checkpoint</p>`;
        $('#log-list').prepend(output);
        phaseButton.disabled = false;
        phaseStatus.textContent = 'Training checkpoint ready. Choose a held-out image.';
        break;
      }
    }
    document.querySelectorAll('[data-client-state]').forEach((element) => {
      element.textContent = 'Ready';
      element.style.color = '';
      element.style.background = '';
    });
  } catch (error) {
    runLabel.textContent = 'Run local model';
    $('#run-state').textContent = 'FAILED';
    const row = document.createElement('div');
    row.className = 'log-item';
    const title = document.createElement('b');
    title.textContent = 'Training error';
    const details = document.createElement('p');
    details.textContent = error.message;
    row.append(title, details);
    $('#log-list').prepend(row);
  } finally {
    datasetInput.disabled = document.querySelectorAll('#dataset-select option').length < 2;
    document.querySelectorAll('[data-start-run]').forEach((button) => { button.disabled = $('#data-status').textContent !== 'LOCAL DATA READY'; });
  }
}

async function loadImagePhases() {
  phaseButton.disabled = true;
  phaseStatus.textContent = 'Loading local image and model activations…';
  try {
    const index = Math.max(0, Number(phaseIndex.value) || 0);
    const response = await fetch(`/api/model/visualization?index=${encodeURIComponent(index)}&dataset_id=${encodeURIComponent(datasetInput.value)}`);
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || 'Could not load model visualizations.');

    const summary = $('#phase-summary');
    summary.textContent = `Image ${result.sample_index + 1} of ${result.sample_count} · True class: ${result.true_class} · Predicted: ${result.predicted_class} · Checkpoint: ${result.checkpoint}`;
    const gallery = $('#phase-gallery');
    gallery.replaceChildren();
    for (const phase of result.phases) {
      const card = document.createElement('article');
      card.className = 'phase-card';
      const heading = document.createElement('h3');
      heading.textContent = `${phase.step}. ${phase.title}`;
      const image = document.createElement('img');
      image.src = phase.image;
      image.alt = phase.title;
      const caption = document.createElement('p');
      caption.textContent = phase.caption;
      card.append(heading, image, caption);
      gallery.append(card);
    }
    $('#phase-result').hidden = false;
    $('#phase-result').style.display = 'block';
    phaseStatus.textContent = 'Visualizations are generated locally from your held-out test image.';
  } catch (error) {
    phaseStatus.textContent = error.message;
  } finally {
    phaseButton.disabled = false;
  }
}

roundInput.addEventListener('input', () => { $('#rounds-value').textContent = roundInput.value; });
epochsInput.addEventListener('input', () => { $('#epochs-value').textContent = epochsInput.value; });
clipInput.addEventListener('input', () => { $('#clip-value').textContent = Number(clipInput.value).toFixed(1); });
datasetInput.addEventListener('change', () => {
  localStorage.setItem('medfedai-dataset', datasetInput.value);
  loadOverview();
});
folderInput.addEventListener('change', describeSelectedFolder);
importButton.addEventListener('click', importSelectedDataset);
noiseInput.addEventListener('input', () => { $('#noise-value').textContent = Number(noiseInput.value).toFixed(1); });
document.querySelectorAll('[data-open-page]').forEach((button) => button.addEventListener('click', () => { window.location.hash = button.dataset.openPage; }));
window.addEventListener('hashchange', showPage);
$('#export-log').addEventListener('click', () => {
  if (!history.length) {
    $('#run-state').textContent = 'NO RESULTS TO EXPORT';
    return;
  }
  const columns = ['round', 'client_count', 'test_accuracy', 'macro_f1', 'test_loss', 'epsilon', 'delta'];
  const lines = [columns.join(','), ...history.map((row) => columns.map((key) => JSON.stringify(row[key] ?? '')).join(','))];
  const blob = new Blob([lines.join('\n')], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `medfedai-${datasetInput.value}-metrics.csv`;
  link.click();
  URL.revokeObjectURL(url);
});
document.querySelectorAll('[data-start-run]').forEach((button) => button.addEventListener('click', () => {
  if (button.id === 'start-run-privacy') window.location.hash = 'overview';
  runModel();
}));
$('#privacy-info').addEventListener('click', () => $('#privacy-dialog').showModal());
$('#dialog-close').addEventListener('click', () => $('#privacy-dialog').close());
$('#dialog-ok').addEventListener('click', () => $('#privacy-dialog').close());
phaseButton.addEventListener('click', loadImagePhases);
window.addEventListener('resize', drawChart);
loadOverview();
showPage();
