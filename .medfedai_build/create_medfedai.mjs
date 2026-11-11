import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { pathToFileURL } from 'node:url';

process.env.RUNTIME_NODE_MODULES = 'C:/Users/pavan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES = process.env.RUNTIME_NODE_MODULES;

const workspaceDir = 'C:/Users/pavan/OneDrive/Project sem 7th/Capstone Project';
const skillDir = 'C:/Users/pavan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const buildDir = path.join(workspaceDir, '.medfedai_build');
const renderDir = path.join(buildDir, 'renders');
const finalPath = path.join(workspaceDir, 'deliverables', 'MedFedAI_Privacy_Focused_Review_II_PrivacyOnly_Final.pptx');
const W = 1280;
const H = 720;
const C = {
  navy: '#10263D',
  navy2: '#183A58',
  teal: '#18A6A6',
  tealLight: '#D9F0EF',
  ink: '#182B3A',
  muted: '#526477',
  pale: '#F4F7F9',
  line: '#D5E0E7',
  white: '#FFFFFF',
  orange: '#C9783A',
};

await fs.mkdir(renderDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });

const pres = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, text, x, y, w, h, size = 24, color = C.ink, bold = false, alignment = 'left') {
  const s = slide.shapes.add({
    geometry: 'textbox',
    position: { left: x, top: y, width: w, height: h },
    fill: 'none',
    line: { style: 'solid', fill: 'none', width: 0 },
  });
  s.text = text;
  s.text.style = { fontFamily: 'Arial', fontSize: size, color, bold, alignment };
  return s;
}

function addRect(slide, x, y, w, h, fill, stroke = fill, strokeWidth = 0, geometry = 'rect') {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: 'solid', fill: stroke, width: strokeWidth },
  });
}

function baseSlide(title, n, source = '') {
  const slide = pres.slides.add();
  slide.background.fill = C.white;
  addText(slide, title, 72, 42, 1120, 58, 40, C.navy, true);
  addRect(slide, 72, 116, 90, 5, C.teal);
  addText(slide, 'MedFedAI  |  Privacy-focused review', 72, 678, 690, 20, 14, C.muted, false);
  addText(slide, String(n).padStart(2, '0'), 1150, 676, 55, 22, 15, C.muted, true, 'right');
  if (source) addText(slide, source, 72, 648, 1125, 22, 13, C.muted, false);
  return slide;
}

// 1. Cover
{
  const s = pres.slides.add();
  s.background.fill = C.navy;
  addRect(s, 0, 0, 18, H, C.teal);
  addText(s, 'CAPSTONE PROJECT I  |  7CS1999', 82, 56, 1000, 30, 18, '#A8D9DB', true);
  addText(s, 'MedFedAI', 82, 116, 1040, 82, 58, C.white, true);
  addText(s, 'Federated intelligence for privacy-preserving\nmedical image classification', 82, 204, 1070, 94, 34, C.white, false);
  addRect(s, 82, 326, 500, 3, C.teal);
  addText(s, 'REVIEW II  |  BATCH ASAC-CP-080', 82, 350, 580, 28, 19, '#A8D9DB', true);
  addText(s, 'Pudi Pavani  ·  2023BCSE07AED340\nKurra Pothula Bhanu Gowtham  ·  2023BCSE07AED344\nSirivela Yashwanth Kumar  ·  2023BCSE07AED347', 82, 402, 740, 106, 18, C.white, false);
  addText(s, 'Alliance School of Advanced Computing  |  CSE-AIML\nMentor: Dr. Smitha Rajagopal  |  Review date: 16 October 2026', 82, 584, 900, 54, 17, '#D4E0E9', false);
}

// 2. Focus and scope
{
  const s = baseSlide('Project focus and scope', 2);
  addText(s, 'One research focus', 82, 152, 390, 34, 25, C.teal, true);
  addText(s, 'Measure and improve patient privacy during federated medical image training.', 82, 194, 1090, 68, 34, C.navy, true);
  addRect(s, 82, 292, 520, 1, C.line);
  addRect(s, 640, 292, 1, 270, C.line);
  addText(s, 'Primary outcome', 82, 314, 450, 34, 24, C.navy, true);
  addText(s, 'How much information can an attacker recover from client model updates, and how does a privacy mechanism change that risk?', 82, 360, 480, 140, 23, C.ink, false);
  addText(s, 'Privacy measures', 690, 314, 460, 34, 24, C.navy, true);
  addText(s, 'Gradient-inversion attack success\n\nReconstruction quality\n\nClient-level privacy budget (ε, δ)', 690, 360, 480, 150, 22, C.ink, false);
  addText(s, 'Case study: brain-tumor MRI classification, aligned with the selected base paper.', 82, 580, 1090, 34, 20, C.muted, false);
}

// 3. Problem statement and threat model
{
  const s = baseSlide('Privacy problem and threat model', 3, 'Privacy risk from shared gradients was demonstrated by Zhu, Liu and Han (2019), “Deep Leakage from Gradients.”');
  addText(s, 'Federated learning keeps raw images at each client, but training still sends model updates to an aggregator.', 82, 156, 1080, 76, 28, C.navy, true);
  addText(s, 'Those updates can expose information about local training examples. The project will test this risk instead of treating data locality as a complete privacy guarantee.', 82, 250, 1080, 86, 23, C.ink, false);
  addText(s, 'Evaluation threat model', 82, 376, 330, 34, 24, C.teal, true);
  addText(s, 'A curious server observes client updates and attempts to reconstruct training information. The evaluation uses a defined gradient-inversion attack on the agreed test setup.', 82, 420, 1070, 74, 22, C.ink, false);
  addRect(s, 82, 548, 10, 72, C.teal);
  addText(s, 'Problem statement', 112, 548, 250, 28, 20, C.navy, true);
  addText(s, 'MedFedAI needs a measurable privacy evaluation for federated brain-tumor MRI classification, with a tested protection mechanism and a reported privacy–utility trade-off.', 112, 582, 1035, 54, 20, C.ink, false);
}

// 4. Literature review
{
  const s = baseSlide('Literature review: privacy lens', 4);
  const rows = [
    {
      y: 150,
      title: 'Rajagopal et al. (2025)  |  FL + Deep CapsNet',
      body: 'Brain-tumor MRI pipeline uses U-Net features, local capsule training and FedAvg. This prior work motivates the architecture; MedFedAI will focus on measuring privacy leakage from shared updates.',
    },
    {
      y: 304,
      title: 'Zhu, Liu & Han (2019)  |  Gradient leakage',
      body: 'Deep Leakage from Gradients shows that shared gradients can reveal private training examples, motivating an attack-based privacy evaluation.',
    },
    {
      y: 458,
      title: 'Zhang et al. (2022)  |  Client-level differential privacy',
      body: 'Studies update clipping and client-level DP for federated learning. It provides a basis for testing a privacy mechanism alongside FedAvg.',
    },
  ];
  for (const row of rows) {
    addText(s, row.title, 82, row.y, 1090, 32, 23, C.navy, true);
    addText(s, row.body, 82, row.y + 42, 1090, 82, 20, C.ink, false);
    if (row.y < 450) addRect(s, 82, row.y + 135, 1090, 1, C.line);
  }
  addText(s, 'Sources: Rajagopal et al. (2025); Zhu et al. (NeurIPS 2019); Zhang et al. (ICML 2022).', 82, 622, 1090, 20, 14, C.muted, false);
}

// 5. Research gap
{
  const s = baseSlide('Research gap and proposed contribution', 5);
  addText(s, 'What prior work establishes', 82, 156, 480, 34, 24, C.navy, true);
  addText(s, 'Federated training can keep raw images on local clients. The selected brain-tumor study reports classification results using FL and Deep CapsNet.', 82, 200, 470, 112, 22, C.ink, false);
  addRect(s, 620, 156, 1, 214, C.line);
  addText(s, 'What this project will measure', 674, 156, 500, 34, 24, C.navy, true);
  addText(s, 'Privacy leakage from transmitted client updates under a stated attack model, plus the effect of client-level differential privacy on leakage and model utility.', 674, 200, 500, 112, 22, C.ink, false);
  addRect(s, 82, 370, 1090, 2, C.teal);
  addText(s, 'Proposed contribution', 82, 400, 260, 30, 22, C.teal, true);
  addText(s, 'A privacy-centered evaluation of MedFedAI: baseline FedAvg, attack-based leakage testing, and a DP-protected comparison with privacy accounting.', 82, 442, 1090, 74, 24, C.navy, true);
  addText(s, 'Privacy leakage and the measured protection level define the project outcome.', 82, 566, 1090, 30, 20, C.muted, false);
}

// 6. Aim, question, objectives
{
  const s = baseSlide('Aim and objectives', 6);
  addText(s, 'Aim', 82, 150, 120, 30, 22, C.teal, true);
  addText(s, 'Evaluate privacy leakage in federated medical image classification and test client-level differential privacy as a protection mechanism.', 82, 186, 1090, 62, 27, C.navy, true);
  addText(s, 'Research question', 82, 278, 260, 30, 22, C.teal, true);
  addText(s, 'How much patient information can be inferred from shared updates, and how does client-level DP change leakage and classification utility?', 82, 314, 1090, 58, 24, C.ink, false);
  addText(s, 'Objectives', 82, 400, 180, 30, 22, C.teal, true);
  addText(s, '1. Build the Deep CapsNet + FedAvg baseline with raw images kept at simulated clients.\n2. Apply update clipping and calibrated noise for client-level DP, and record ε and δ.\n3. Test baseline and protected updates with a gradient-inversion attack.\n4. Compare attack success and reconstruction quality before and after DP.', 82, 442, 1100, 170, 21, C.ink, false);
}

// 7. Requirements and constraints
{
  const s = baseSlide('Requirements and constraints', 7);
  addText(s, 'Data and software', 82, 152, 430, 32, 24, C.navy, true);
  addText(s, 'Brain-tumor MRI images with class labels\nSimulated hospital clients with train/test partitions\nPython and PyTorch or TensorFlow\nFlower or TensorFlow Federated for FL\nGPU/CPU, storage and reproducible experiment logs', 82, 198, 500, 210, 21, C.ink, false);
  addRect(s, 630, 152, 1, 382, C.line);
  addText(s, 'Privacy and implementation constraints', 680, 152, 500, 32, 24, C.navy, true);
  addText(s, 'Medical images remain local in the simulation.\n\nDP noise can affect training utility and convergence.\n\nA privacy claim requires a defined threat model and correct privacy accounting.\n\nCompute limits may constrain client count and attack experiments.', 680, 198, 500, 285, 21, C.ink, false);
  addText(s, 'Simulated hospital clients provide the setting for privacy testing. They are not a separate research objective.', 82, 574, 1090, 44, 20, C.muted, false);
}

// 8. Architecture diagram
{
  const s = baseSlide('MedFedAI architecture', 8, 'The proposed differential-privacy mechanism and attack evaluation extend the FL + Deep CapsNet baseline described in the supplied project material.');
  const y = 246;
  const h = 126;
  const boxes = [
    { x: 72, w: 190, title: 'Client data', sub: 'MRI stays local' },
    { x: 300, w: 190, title: 'Local model', sub: 'Deep CapsNet' },
    { x: 528, w: 190, title: 'Privacy step', sub: 'Clip + calibrated noise' },
    { x: 756, w: 190, title: 'Aggregation', sub: 'Server-side FedAvg' },
    { x: 984, w: 220, title: 'Global model', sub: 'Broadcast to clients' },
  ];
  const shapes = [];
  for (const b of boxes) {
    const r = addRect(s, b.x, y, b.w, h, C.pale, C.line, 2, 'roundRect');
    shapes.push(r);
    addText(s, b.title, b.x + 14, y + 22, b.w - 28, 32, 22, C.navy, true, 'center');
    addText(s, b.sub, b.x + 14, y + 66, b.w - 28, 44, 18, C.muted, false, 'center');
  }
  for (let i = 0; i < shapes.length - 1; i++) {
    s.shapes.connect(shapes[i], shapes[i + 1], { kind: 'straight', fromSide: 'right', toSide: 'left', line: { style: 'solid', fill: C.teal, width: 3 }, tail: { type: 'arrow', width: 'med', length: 'med' } });
  }
  addText(s, 'Privacy audit', 82, 442, 220, 30, 22, C.teal, true);
  addText(s, 'Gradient-inversion attack against exposed updates\nPrivacy accountant reports ε and δ for the DP training run', 82, 482, 680, 76, 21, C.ink, false);
  addText(s, 'Privacy accounting', 820, 442, 270, 30, 22, C.teal, true);
  addText(s, 'Report the client-level privacy budget (ε, δ) for the protected training run.', 820, 482, 360, 76, 21, C.ink, false);
}

// 9. Stepwise methodology
{
  const s = baseSlide('Proposed methodology', 9);
  const steps = [
    ['01', 'Prepare the MRI dataset', 'Clean labels, normalize images and create a held-out test set.'],
    ['02', 'Simulate client sites', 'Distribute training data across the simulated hospital clients.'],
    ['03', 'Train the FL baseline', 'Train Deep CapsNet locally and aggregate updates with FedAvg.'],
    ['04', 'Measure baseline leakage', 'Run a defined gradient-inversion attack against shared updates.'],
    ['05', 'Add client-level DP', 'Clip client updates, add calibrated noise and track the privacy budget.'],
    ['06', 'Compare privacy', 'Compare attack success and reconstruction quality before and after DP.'],
  ];
  let y = 145;
  for (const [num, title, body] of steps) {
    addText(s, num, 82, y, 58, 38, 22, C.teal, true);
    addText(s, title, 156, y, 320, 32, 22, C.navy, true);
    addText(s, body, 490, y, 690, 44, 19, C.ink, false);
    if (num !== '06') addRect(s, 156, y + 54, 1024, 1, C.line);
    y += 82;
  }
  addText(s, 'The defense mechanism and accounting parameters must be fixed before final experiments.', 82, 650, 1090, 20, 15, C.muted, false);
}

// 10. Evaluation plan
{
  const s = baseSlide('Evaluation plan', 10);
  addText(s, 'Privacy outcomes', 82, 154, 420, 34, 24, C.navy, true);
  addText(s, 'Gradient-inversion attack success\nReconstruction similarity or recovery quality\nClient-level privacy budget (ε, δ)', 82, 202, 500, 132, 22, C.ink, false);
  addRect(s, 630, 154, 1, 234, C.line);
  addText(s, 'Privacy comparison', 680, 154, 450, 34, 24, C.navy, true);
  addText(s, 'Baseline FedAvg vs DP-protected FedAvg\nSame data, model, client setup and attack settings', 680, 202, 490, 108, 22, C.ink, false);
  addRect(s, 82, 388, 1090, 2, C.teal);
  addText(s, 'Interpretation', 82, 418, 240, 30, 22, C.teal, true);
  addText(s, 'Report how much the selected DP mechanism changes attack outcomes and state the privacy guarantee with its accounting assumptions.', 82, 460, 1080, 72, 23, C.navy, true);
  addText(s, 'No implementation results were included in the supplied material. This slide defines planned measurements only.', 82, 574, 1090, 34, 19, C.muted, false);
}

// 11. References
{
  const s = baseSlide('References', 11);
  const refs = [
    'Rajagopal, S., Shieh, C.-S., Shankar, S. S., Maithili, K., & Chakrabarti, P. (2025). Privacy-Preserving Multi-Class Brain Tumor Classification Using Federated Learning and Deep Capsule Networks. Architecture Image Studies, 6(3), 1677–1699. https://doi.org/10.62754/ais.v6i3.504',
    'McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, PMLR 54, 1273–1282.',
    'Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. Advances in Neural Information Processing Systems 32.',
    'Zhang, X., Chen, X., Hong, M., Wu, S., & Yi, J. (2022). Understanding Clipping for Federated Learning: Convergence and Client-Level Differential Privacy. ICML, PMLR 162, 26048–26067.',
  ];
  let y = 154;
  for (const ref of refs) {
    addRect(s, 82, y + 3, 5, 82, C.teal);
    addText(s, ref, 108, y, 1070, 94, 18, C.ink, false);
    y += 116;
  }
  addText(s, 'The selected base paper motivates the FL and Deep CapsNet setup. This project evaluates privacy.', 82, 626, 1090, 28, 15, C.muted, false);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

for (const [index, slide] of pres.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, '0')}`;
  await writeBlob(path.join(renderDir, `${stem}.png`), await pres.export({ slide, format: 'png', scale: 1 }));
}
await writeBlob(path.join(renderDir, 'montage.webp'), await pres.export({ format: 'webp', montage: true }));

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidatePath);
const result = await finalizePresentation({
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: 'C:/Users/pavan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  fontPolicy: { basis: 'design', families: ['Arial'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation-privacy-only-final.json'),
});
console.log(JSON.stringify({ finalPath, slideCount: pres.slides.items.length, result }, null, 2));
