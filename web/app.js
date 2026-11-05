'use strict';

/* =========================================================================
 * GigaAM Web UI — логика интерфейса (ванильный JS, без зависимостей)
 * ========================================================================= */

const $ = (id) => document.getElementById(id);

const LS_KEY = 'gigaam.webui.job';
const LS_PROMPT_KEY = 'gigaam.webui.sysprompt';
const LS_MODEL_KEY = 'gigaam.webui.llmmodel';
const LS_SUMMARY_KEY = 'gigaam.webui.summary';
const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
const STATUS_RU = {
  queued: 'В очереди',
  running: 'Выполняется',
  completed: 'Завершена',
  failed: 'Ошибка',
  cancelled: 'Отменена',
};

// Должен совпадать с DEFAULT_SYSTEM_PROMPT в web/app.py.
const DEFAULT_SYSTEM_PROMPT = [
  'Ты — ассистент, который суммаризирует расшифровки встреч (митингов) на русском языке. Тебе даётся стенограмма с метками времени и спикеров.',
  '',
  'Составь структурированное резюме встречи:',
  '1. Краткое описание (2–4 предложения): о чём встреча и кто участвовал.',
  '2. Основные темы обсуждения — короткий маркированный список.',
  '3. Принятые решения — список; если решений нет, так и напиши.',
  '4. Открытые вопросы и неясности.',
  '5. Action items (задачи): кто — что делает — к какому сроку; если срок не прозвучал, укажи «срок не назван».',
  '',
  'Правила:',
  '- Опирайся только на текст расшифровки, ничего не додумывай.',
  '- Сохраняй обозначения спикеров как в стенограмме (SPEAKER_XX), если настоящие имена не прозвучали.',
  '- Пиши кратко и по делу, без воды.',
  '- Отвечай на русском языке; если стенограмма на другом языке — отвечай на языке стенограммы.',
  '- Оформи ответ в Markdown с заголовками разделов.',
].join('\n');

const VIDEO_EXT = ['mp4', 'mkv', 'mov', 'webm', 'avi', 'm4v', 'mpg', 'mpeg', 'ts', '3gp'];
const AUDIO_EXT = ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'oga', 'opus', 'wma', 'aiff', 'aif'];

/* ========================= состояние ========================= */

const state = {
  file: null,          // выбранный File
  fileUrl: null,       // object URL для локального предпросмотра
  fileKind: null,      // 'audio' | 'video'
  duration: null,      // длительность из метаданных (может быть null)
  job: null,           // последняя отслеживаемая задача (JobResponse)
  result: null,        // результат последней завершённой задачи
  jobFileBase: '',     // имя файла без расширения — для имён выгрузок
  pollTimer: null,
  pollJobId: null,
  pollFails: 0,
  elapsedTimer: null,
  elapsedBase: null,
  summarizing: false,  // выполняется запрос /api/summarize
  summaryFor: null,    // job_id, для которого получена суммаризация
  summaryData: null,   // ответ /api/summarize {summary, model, usage}
};

/* ========================= утилиты ========================= */

function escapeHtml(s) {
  // Сущности собираются конкатенацией: литерал "&amp;" в исходнике
  // рискует быть нормализован обратно в "&" при записи файла.
  const amp = '&' + '';
  const map = {
    '&': amp + 'amp;',
    '<': amp + 'lt;',
    '>': amp + 'gt;',
    '"': amp + 'quot;',
    "'": amp + '#39;',
  };
  return String(s ?? '').replace(/[&<>"']/g, (c) => map[c]);
}

function pad2(n) { return String(n).padStart(2, '0'); }

/** 78.4 -> «1:18» */
function fmtClock(sec) {
  if (sec == null || !isFinite(sec)) return '—';
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${pad2(m)}:${pad2(s)}` : `${m}:${pad2(s)}`;
}

/** 78.456 -> «00:01:18,456» (для SRT) */
function fmtSrt(sec) {
  if (sec == null || !isFinite(sec)) sec = 0;
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec - Math.floor(sec)) * 1000);
  return `${pad2(h)}:${pad2(m)}:${pad2(s)},${String(ms).padStart(3, '0')}`;
}

/** секунд, прошедших с baseMs -> «12:34» / «1:02:03» */
function fmtDuration(sec) { return fmtClock(sec); }

function fmtBytes(n) {
  if (!isFinite(n) || n < 0) return '';
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} ГБ`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} МБ`;
  return `${Math.round(n / 1024)} КБ`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return '—';
  return d.toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', year: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** стабильный «цвет» спикера по метке */
function speakerHue(label) {
  let h = 0;
  for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}
function speakerColor(label) { return `hsl(${speakerHue(label)}, 75%, 72%)`; }
function chipStyle(label) {
  const h = speakerHue(label);
  return `color:hsl(${h},75%,72%);background:hsla(${h},70%,60%,.16);border-color:hsla(${h},70%,60%,.45)`;
}

/* ========================= обёртка API ========================= */

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch('/api' + path, options);
  } catch (e) {
    throw { status: 0, detail: 'Сервер недоступен' };
  }
  let body = null;
  try { body = await res.json(); } catch { /* не JSON */ }
  if (!res.ok) {
    throw { status: res.status, detail: (body && body.detail) || `Ошибка ${res.status}` };
  }
  return body;
}

/* ========================= показ ошибок ========================= */

function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove('hidden');
}
function hideErrors() {
  for (const id of ['form-error', 'job-error']) $(id).classList.add('hidden');
}

/* ========================= файл и предпросмотр ========================= */

function fileKindOf(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (VIDEO_EXT.includes(ext)) return 'video';
  if (AUDIO_EXT.includes(ext)) return 'audio';
  if (file.type.startsWith('video/')) return 'video';
  return 'audio';
}

function probeDuration(file, kind) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const el = document.createElement(kind === 'video' ? 'video' : 'audio');
    const done = (value) => {
      clearTimeout(timer);
      el.onloadedmetadata = el.onerror = null;
      URL.revokeObjectURL(url);
      resolve(value);
    };
    const timer = setTimeout(() => done(null), 5000);
    el.onloadedmetadata = () => done(isFinite(el.duration) ? el.duration : null);
    el.onerror = () => done(null);
    el.preload = 'metadata';
    el.src = url;
  });
}

async function setFile(file) {
  if (!file) return;
  state.file = file;
  state.fileKind = fileKindOf(file);
  state.duration = null;
  state.result = null;
  hideErrors();
  // новый файл — прежняя суммаризация больше не относится к задаче
  state.summarizing = false;
  state.summaryFor = null;
  state.summaryData = null;
  resetSummaryUI();

  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.fileUrl = URL.createObjectURL(file);
  state.jobFileBase = file.name.replace(/\.[^.]+$/, '') || 'result';

  const badge = $('file-kind');
  badge.textContent = state.fileKind === 'video' ? 'видео' : 'аудио';
  badge.className = 'badge ' + state.fileKind;
  $('file-name').textContent = file.name;
  $('file-meta').textContent = `${fmtBytes(file.size)} · определение длительности…`;

  $('file-info').classList.remove('hidden');
  $('result-empty').classList.remove('hidden');
  $('result').classList.add('hidden');

  const video = $('preview-video'), audio = $('preview-audio');
  video.classList.toggle('hidden', state.fileKind !== 'video');
  audio.classList.toggle('hidden', state.fileKind !== 'audio');
  const el = state.fileKind === 'video' ? video : audio;
  el.src = state.fileUrl;

  const dur = await probeDuration(file, state.fileKind);
  state.duration = dur;
  if (state.file === file) {
    $('file-meta').textContent =
      `${fmtBytes(file.size)}${dur != null ? ' · ' + fmtClock(dur) : ''}`;
  }

  enableSubmit();
}

function clearFile() {
  state.file = null;
  state.duration = null;
  if (state.fileUrl) { URL.revokeObjectURL(state.fileUrl); state.fileUrl = null; }
  $('file-input').value = '';
  $('file-info').classList.add('hidden');
  $('preview-video').classList.add('hidden');
  $('preview-audio').classList.add('hidden');
  $('preview-video').removeAttribute('src');
  $('preview-audio').removeAttribute('src');
  enableSubmit();
}

function enableSubmit() {
  $('submit-btn').disabled = !state.file;
}

/* ========================= параметры спикеров ========================= */

function syncSpeakerFields(changed) {
  const num = $('num-speakers'), min = $('min-speakers'), max = $('max-speakers');
  if (changed === num) {
    if (num.value !== '') { min.value = ''; max.value = ''; }
    min.disabled = num.value !== '';
    max.disabled = num.value !== '';
  } else {
    if (min.value !== '' || max.value !== '') num.value = '';
    num.disabled = min.value !== '' || max.value !== '';
  }
}

function buildParams() {
  const p = new URLSearchParams();
  const errors = [];
  const num = $('num-speakers').value.trim();
  const min = $('min-speakers').value.trim();
  const max = $('max-speakers').value.trim();
  const ov = $('min-overlap').value.trim();

  const intField = (raw, label) => {
    if (raw === '') return null;
    const v = Number(raw);
    if (!Number.isInteger(v) || v < 1) { errors.push(`${label} — целое число ≥ 1`); return null; }
    return v;
  };

  const numV = intField(num, 'Точное число спикеров');
  if (numV != null) p.set('num_speakers', numV);
  const minV = intField(min, 'Минимум спикеров');
  if (minV != null) p.set('min_speakers', minV);
  const maxV = intField(max, 'Максимум спикеров');
  if (maxV != null) p.set('max_speakers', maxV);
  if (minV != null && maxV != null && minV > maxV) {
    errors.push('Минимум спикеров не может быть больше максимума');
  }
  if (ov !== '') {
    const v = Number(ov);
    if (isNaN(v) || v < 0 || v > 1) errors.push('Порог совпадения — число от 0 до 1');
    else p.set('min_overlap', String(v));
  }
  return { query: p.toString(), error: errors.join('; ') };
}

/* ========================= прогресс ========================= */

function showProgress() {
  $('progress').classList.remove('hidden');
}
function hideProgress() {
  $('progress').classList.add('hidden');
  stopElapsed();
}
function setBar(percent) {
  const bar = document.querySelector('.bar');
  const fill = $('bar-fill');
  if (percent == null) {
    bar.classList.add('indeterminate');
    fill.style.width = '';
  } else {
    bar.classList.remove('indeterminate');
    fill.style.width = `${percent}%`;
  }
}
function setStatus(text, spinner = true) {
  $('status-label').textContent = text;
  document.querySelector('.spinner').style.display = spinner ? '' : 'none';
}
function startElapsed(baseMs) {
  stopElapsed();
  state.elapsedBase = baseMs;
  updateElapsed();
  state.elapsedTimer = setInterval(updateElapsed, 1000);
}
function stopElapsed() {
  if (state.elapsedTimer) { clearInterval(state.elapsedTimer); state.elapsedTimer = null; }
  state.elapsedBase = null;
  $('elapsed').textContent = '';
}
function updateElapsed() {
  if (state.elapsedBase == null) return;
  const sec = Math.max(0, (Date.now() - state.elapsedBase) / 1000);
  $('elapsed').textContent = `${fmtDuration(sec)}${sec < 90 ? ' с' : ''}`;
}

/* ========================= отправка ========================= */

function uploadFile(file, query) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/transcribe_diarized' + (query ? `?${query}` : ''));
    xhr.responseType = 'json';
    setBar(0);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setBar(Math.round((e.loaded / e.total) * 100));
    };
    xhr.upload.onload = () => {
      setBar(null);
      setStatus('Файл передан — сервер сохраняет и ставит задачу в очередь');
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject({
        status: xhr.status,
        detail: (xhr.response && xhr.response.detail) || `Ошибка ${xhr.status}`,
      });
    };
    xhr.onerror = () => reject({ status: 0, detail: 'Сеть недоступна' });
    xhr.ontimeout = () => reject({ status: 0, detail: 'Превышено время ожидания' });

    const fd = new FormData();
    fd.append('file', file, file.name);
    xhr.send(fd);
  });
}

async function onSubmit() {
  if (!state.file) return;
  hideErrors();

  const { query, error } = buildParams();
  if (error) { showError($('form-error'), error); return; }

  // Сервер обрабатывает одну задачу за раз: предупредим заранее, а не после загрузки.
  try {
    const data = await api('/jobs');
    const jobs = Array.isArray(data) ? data : (data && data.jobs) || [];
    const active = jobs.find((j) => j.status === 'queued' || j.status === 'running');
    if (active && !confirm(
      'На сервере уже есть активная задача (ID ' + String(active.job_id).slice(0, 8) +
      ', ' + (STATUS_RU[active.status] || active.status) + ').\n' +
      'Одновременная обработка не поддерживается: новая задача получит ошибку 409, ' +
      'пока предыдущая не завершится.\n\nВсё равно отправить?'
    )) return;
  } catch (e) { /* сервер недоступен — ошибка проявится при отправке */ }

  const btn = $('submit-btn');
  btn.disabled = true;
  btn.textContent = 'Отправка…';
  showProgress();
  setBar(0);
  setStatus('Загрузка файла…');
  startElapsed(Date.now());
  $('cancel-btn').classList.add('hidden');

  try {
    const job = await uploadFile(state.file, query);
    state.job = job;
    state.result = null;
    saveLastJob();
    setStatus(STATUS_RU[job.status] || job.status);
    startElapsed(Date.parse(job.started_at || job.created_at || Date.now()));
    refreshHistory();
    startPolling(job.job_id);
  } catch (e) {
    hideProgress();
    showError($('job-error'), e.detail || 'Не удалось отправить файл');
    enableSubmit();
  } finally {
    btn.textContent = 'Отправить на диаризацию';
  }
}

/* ========================= опрос статуса ========================= */

function startPolling(jobId) {
  stopPolling();
  state.pollJobId = jobId;
  state.pollFails = 0;
  showProgress();
  setBar(null);
  pollTick();
}

function stopPolling() {
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  state.pollJobId = null;
}

function pollTick() {
  state.pollTimer = setTimeout(async () => {
    const jobId = state.pollJobId;
    if (!jobId) return;
    try {
      const job = await api(`/jobs/${jobId}`);
      state.pollFails = 0;
      state.job = job;
      renderJobStatus(job);
      if (TERMINAL.has(job.status)) { finishJob(job); return; }
      pollTick();
    } catch (err) {
      if (err.status === 404) {
        stopPolling();
        hideProgress();
        showError($('job-error'), 'Задача не найдена — возможно, результат удалён по TTL или задача была отменена.');
        clearLastJob();
        enableSubmit();
        refreshHistory();
        return;
      }
      state.pollFails += 1;
      if (state.pollFails >= 10) {
        stopPolling();
        showError($('job-error'), 'Сервер недоступен, опрос остановлен. Откройте задачу из истории позже.');
        return;
      }
      pollTick();
    }
  }, 2000);
}

function renderJobStatus(job) {
  setStatus(STATUS_RU[job.status] || job.status, job.status !== 'completed');
  const base = Date.parse(job.started_at || job.created_at || '');
  if (!isNaN(base)) startElapsed(base);
  $('cancel-btn').classList.toggle('hidden', job.status !== 'queued');
}

function finishJob(job) {
  stopPolling();
  stopElapsed();
  $('cancel-btn').classList.add('hidden');
  refreshHistory();
  enableSubmit();

  if (job.status === 'completed') {
    hideProgress();
    clearLastJob();
    renderResult(job.result || {});
    // Автосуммаризация — только здесь: свежее завершение задачи.
    // restoreLastJob/openJob просто перерисовывают результат и не тратят токены.
    requestSummary('auto');
  } else if (job.status === 'failed') {
    hideProgress();
    showError($('job-error'), job.error || 'Задача завершилась с ошибкой');
    clearLastJob();
  } else {
    hideProgress();
    showError($('job-error'), 'Задача отменена.');
    clearLastJob();
  }
}

async function onCancel() {
  const job = state.job;
  if (!job) return;
  const btn = $('cancel-btn');
  btn.disabled = true;
  try {
    await api(`/jobs/${job.job_id}`, { method: 'DELETE' });
    // опрос подхватит статус cancelled
  } catch (err) {
    showError($('job-error'), err.detail || 'Не удалось отменить задачу');
  } finally {
    btn.disabled = false;
  }
}

/* ========================= результат ========================= */

function showEmptyResult() {
  $('result-empty').classList.remove('hidden');
  $('result').classList.add('hidden');
  state.result = null;
  resetSummaryUI();
}

function renderResult(result) {
  state.result = result;
  $('result-empty').classList.add('hidden');
  $('result').classList.remove('hidden');

  const parts = [];
  if (result.duration != null) parts.push(`Длительность: ${fmtClock(result.duration)}`);
  if (result.num_speakers != null) parts.push(`Спикеров: ${result.num_speakers}`);
  if (state.job && state.job.job_id) parts.push(`Задача: ${String(state.job.job_id).slice(0, 8)}`);
  $('result-meta').textContent = parts.join(' · ');

  const chips = $('speakers-chips');
  chips.innerHTML = '';
  for (const sp of result.speakers || []) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = sp;
    chip.setAttribute('style', chipStyle(sp));
    chips.appendChild(chip);
  }

  const dlg = $('dialog-view');
  dlg.innerHTML = '';
  const utts = result.utterances || [];
  if (!utts.length) {
    dlg.innerHTML = '<div class="muted empty">Речь не обнаружена</div>';
  }
  for (const u of utts) {
    const row = document.createElement('div');
    row.className = 'utt';
    row.style.borderLeftColor = speakerColor(u.speaker);

    const time = document.createElement('button');
    time.type = 'button';
    time.className = 'utt-time';
    time.textContent = fmtClock(u.start);
    time.title = `${fmtClock(u.start)} – ${fmtClock(u.end)} — перейти во время`;
    time.addEventListener('click', () => seekPreview(u.start));

    const sp = document.createElement('span');
    sp.className = 'utt-speaker';
    sp.textContent = u.speaker;
    sp.style.color = speakerColor(u.speaker);

    const tx = document.createElement('div');
    tx.className = 'utt-text';
    tx.textContent = u.text || '';

    row.append(time, sp, tx);
    dlg.appendChild(row);
  }

  $('text-view').textContent = result.text || '(пусто)';
  switchTab('dialog');
  restoreOrResetSummary();
}

function switchTab(view) {
  for (const btn of document.querySelectorAll('.tab')) {
    btn.classList.toggle('active', btn.dataset.view === view);
  }
  $('dialog-view').classList.toggle('hidden', view !== 'dialog');
  $('text-view').classList.toggle('hidden', view !== 'text');
}

function seekPreview(sec) {
  if (!state.fileUrl) return;
  const el = state.fileKind === 'video' ? $('preview-video') : $('preview-audio');
  if (el.classList.contains('hidden')) return;
  el.currentTime = sec;
  el.play().catch(() => { /* автозапуск может быть запрещён */ });
}

/* ========================= мини-Markdown (безопасный рендер) ========================= */

/** Инлайновая разметка; вход уже экранирован escapeHtml. */
function mdInline(s) {
  return s
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

/**
 * Markdown → HTML. Весь текст экранируется ДО вставки тегов, поэтому
 * HTML/JS из ответа LLM не может «просочиться» (XSS-safe).
 * Поддержка: заголовки, **жирный**, *курсив*, `код`, ```-блоки, списки,
 * таблицы, цитаты, ссылки, горизонтальные линии.
 */
function mdToHtml(md) {
  const lines = String(md ?? '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let para = [];
  let openList = null;   // 'ul' | 'ol'
  let code = null;       // { buf: [] } для ```-блоков
  let quote = [];
  let table = null;      // { header: [], rows: [[]] }

  const closePara = () => {
    if (para.length) { out.push(`<p>${mdInline(para.join('<br>'))}</p>`); para = []; }
  };
  const closeList = () => {
    if (openList) { out.push(`</${openList}>`); openList = null; }
  };
  const closeQuote = () => {
    if (quote.length) { out.push(`<blockquote>${mdToHtml(quote.join('\n'))}</blockquote>`); quote = []; }
  };
  const closeTable = () => {
    if (!table) return;
    let html = '<table><thead><tr>';
    for (const c of table.header) html += `<th>${mdInline(c)}</th>`;
    html += '</tr></thead>';
    if (table.rows.length) {
      html += '<tbody>';
      for (const r of table.rows) {
        html += '<tr>';
        for (const c of r) html += `<td>${mdInline(c)}</td>`;
        html += '</tr>';
      }
      html += '</tbody>';
    }
    out.push(html + '</table>');
    table = null;
  };
  const closeAll = () => { closeTable(); closeQuote(); closeList(); closePara(); };

  const splitRow = (line) => {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split('|').map((c) => c.trim());
  };
  const isSepRow = (cells) =>
    cells.length > 0 && cells.every((c) => /^:?-+:?$/.test(c));

  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx];

    if (code !== null) {
      if (/^\s*```/.test(line)) { out.push(`<pre><code>${escapeHtml(code.buf.join('\n'))}</code></pre>`); code = null; }
      else code.buf.push(line);
      continue;
    }
    const fence = line.match(/^\s*```/);
    if (fence) { closeAll(); code = { buf: [] }; continue; }

    if (!line.trim()) { closeAll(); continue; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      closeAll();
      const lvl = h[1].length;
      out.push(`<h${lvl}>${mdInline(escapeHtml(h[2].trim()))}</h${lvl}>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { closeAll(); out.push('<hr>'); continue; }

    if (/^\s*>/.test(line)) {
      closeTable(); closeList(); closePara();
      quote.push(line.replace(/^\s*>\s?/, ''));
      continue;
    }

    // таблица: строка с «|», за которой следует разделитель из дефисов
    if (line.includes('|') && idx + 1 < lines.length) {
      const next = lines[idx + 1];
      if (next.includes('|') && isSepRow(splitRow(next))) {
        closeAll();
        table = { header: splitRow(line).map(escapeHtml), rows: [] };
        idx++; // пропускаем разделитель
        continue;
      }
    }
    if (table && line.includes('|')) { table.rows.push(splitRow(line).map(escapeHtml)); continue; }

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      closeTable(); closeQuote(); closePara();
      const want = ul ? 'ul' : 'ol';
      if (openList !== want) { closeList(); out.push(`<${want}>`); openList = want; }
      out.push(`<li>${mdInline(escapeHtml((ul || ol)[1].trim()))}</li>`);
      continue;
    }

    para.push(escapeHtml(line.trim()));
  }
  closeAll();
  return out.join('\n');
}

/* ========================= LLM-суммаризация ========================= */

function resetSummaryUI(emptyText) {
  $('summary-progress').classList.add('hidden');
  $('summary-error').classList.add('hidden');
  $('summary-result').classList.add('hidden');
  const empty = $('summary-empty');
  empty.classList.remove('hidden');
  empty.textContent = emptyText ||
    'После завершения диаризации расшифровка автоматически уйдёт в Ollama, и здесь появится резюме встречи.';
  $('summary-rerun').disabled = !state.result;
}

function resultHasText(result) {
  if (!result) return false;
  const utts = Array.isArray(result.utterances) ? result.utterances : [];
  if (utts.some((u) => String((u && u.text) || '').trim())) return true;
  return String(result.text || '').trim().length > 0;
}

function readSummaryCache() {
  try {
    const raw = localStorage.getItem(LS_SUMMARY_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && parsed.jobId && parsed.data) return parsed;
  } catch (err) { /* повреждённый кэш игнорируем */ }
  return null;
}

function saveSummaryCache() {
  try {
    if (state.summaryFor && state.summaryData) {
      localStorage.setItem(LS_SUMMARY_KEY, JSON.stringify({
        jobId: state.summaryFor,
        data: state.summaryData,
      }));
    }
  } catch (err) { /* приватный режим браузера и т.п. */ }
}

function clearSummaryCache() {
  try { localStorage.removeItem(LS_SUMMARY_KEY); } catch (err) { /* ignore */ }
}

function renderSummary(data, startedMs) {
  $('summary-progress').classList.add('hidden');
  $('summary-error').classList.add('hidden');
  $('summary-empty').classList.add('hidden');
  $('summary-result').classList.remove('hidden');

  const parts = [];
  if (data.model) parts.push(`модель: ${data.model}`);
  const usage = data.usage || {};
  const tokens = usage.total_tokens ?? usage.prompt_tokens ?? usage.completion_tokens;
  if (tokens != null) parts.push(`токенов: ${tokens}`);
  if (startedMs != null) parts.push(`${((Date.now() - startedMs) / 1000).toFixed(1)} с`);
  $('summary-meta').textContent = parts.join(' · ');
  $('summary-body').innerHTML = mdToHtml(data.summary || '(пусто)');
}

/** Показывает прежнюю суммаризацию этой же задачи или сбрасывает панель. */
function restoreOrResetSummary() {
  const jobId = state.job && state.job.job_id;
  if (state.summarizing) return; // не трогаем панель во время активного запроса
  if (state.summaryData && state.summaryFor && state.summaryFor === jobId) {
    renderSummary(state.summaryData, null);
    return;
  }
  const cached = readSummaryCache();
  if (cached && cached.jobId === jobId) {
    state.summaryFor = cached.jobId;
    state.summaryData = cached.data;
    renderSummary(cached.data, null);
    return;
  }
  state.summaryFor = null;
  state.summaryData = null;
  resetSummaryUI();
}

async function requestSummary(trigger) {
  if (state.summarizing) return;
  const result = state.result;
  if (!result) return;
  if (!resultHasText(result)) {
    state.summaryFor = null;
    state.summaryData = null;
    resetSummaryUI('Речь в задаче не обнаружена — суммаризировать нечего.');
    return;
  }
  state.summarizing = true;
  $('summary-rerun').disabled = true;
  $('summary-empty').classList.add('hidden');
  $('summary-error').classList.add('hidden');
  $('summary-result').classList.add('hidden');
  $('summary-progress').classList.remove('hidden');
  $('summary-status-label').textContent =
    trigger === 'auto' ? 'Суммаризация встречи…' : 'Повторная суммаризация встречи…';
  const started = Date.now();
  try {
    const payload = {
      utterances: result.utterances || null,
      text: result.text || null,
      duration: result.duration ?? null,
      speakers: result.speakers || null,
      system_prompt: ($('system-prompt').value || '').trim() || null,
      model: ($('llm-model').value || '').trim() || null,
    };
    const data = await api('/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.summaryFor = state.job ? state.job.job_id : null;
    state.summaryData = data;
    saveSummaryCache();
    renderSummary(data, started);
  } catch (err) {
    state.summaryFor = null;
    state.summaryData = null;
    clearSummaryCache();
    $('summary-progress').classList.add('hidden');
    showError($('summary-error'),
      (err && err.detail) ||
      'Не удалось получить суммаризацию. Проверьте подключение и нажмите «Суммаризовать заново».');
  } finally {
    state.summarizing = false;
    $('summary-rerun').disabled = !state.result;
  }
}

/* ========================= экспорт ========================= */

function buildTxt(result) {
  const lines = ['Диаризация диалога — GigaAM'];
  if (state.job && state.job.job_id) lines.push(`Задача: ${state.job.job_id}`);
  lines.push('');
  for (const u of result.utterances || []) {
    lines.push(`[${fmtClock(u.start)} – ${fmtClock(u.end)}] ${u.speaker}: ${u.text}`);
  }
  if (result.text) lines.push('', '────────────────────────', '', result.text);
  return lines.join('\n');
}

function buildSrt(result) {
  return (result.utterances || [])
    .map((u, i) => `${i + 1}\n${fmtSrt(u.start)} --> ${fmtSrt(u.end)}\n${u.speaker}: ${u.text}\n`)
    .join('\n');
}

function download(name, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportBase() {
  return state.jobFileBase || 'result';
}

async function onCopy() {
  if (!state.result) return;
  const text = state.result.text || buildTxt(state.result);
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
  const btn = $('copy-btn');
  const old = btn.textContent;
  btn.textContent = 'Скопировано ✓';
  setTimeout(() => { btn.textContent = old; }, 1500);
}

/* ========================= история задач ========================= */

async function refreshHistory() {
  try {
    const data = await api('/jobs');
    renderHistory(Array.isArray(data) ? data : (data && data.jobs) || []);
  } catch (e) { /* состояние сервера видно по чипу здоровья */ }
}

function renderHistory(jobs) {
  const wrap = $('history-wrap');
  const empty = $('history-empty');
  const tbody = document.querySelector('#history-table tbody');
  tbody.innerHTML = '';
  if (!jobs.length) {
    wrap.classList.add('hidden');
    empty.classList.remove('hidden');
    return;
  }
  empty.classList.add('hidden');
  wrap.classList.remove('hidden');

  const sorted = [...jobs].sort((a, b) =>
    String(b.created_at || '').localeCompare(String(a.created_at || '')));

  for (const job of sorted) {
    const tr = document.createElement('tr');

    const id = document.createElement('td');
    id.className = 'mono';
    id.textContent = String(job.job_id).slice(0, 8);
    id.title = job.job_id;

    const st = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `status-badge ${job.status}`;
    badge.textContent = STATUS_RU[job.status] || job.status;
    st.appendChild(badge);

    const created = document.createElement('td');
    created.textContent = fmtDate(job.created_at);

    const finished = document.createElement('td');
    finished.textContent = fmtDate(job.finished_at);

    const err = document.createElement('td');
    err.className = 'cell-error';
    if (job.error) {
      err.textContent = job.error;
      err.title = job.error;
    }

    const act = document.createElement('td');
    act.className = 'cell-actions';

    const open = document.createElement('button');
    open.className = 'ghost';
    open.textContent = 'Открыть';
    open.addEventListener('click', () => openJob(job.job_id));

    const del = document.createElement('button');
    del.className = 'ghost';
    del.textContent = 'Удалить';
    del.addEventListener('click', () => deleteJob(job.job_id));

    act.append(open, del);
    tr.append(id, st, created, finished, err, act);
    tbody.appendChild(tr);
  }
}

async function deleteJob(id) {
  if (!confirm(`Удалить задачу ${String(id).slice(0, 8)} с сервера?`)) return;
  try {
    await api(`/jobs/${id}`, { method: 'DELETE' });
    refreshHistory();
  } catch (err) {
    alert(err.detail || 'Не удалось удалить задачу');
  }
}

async function openJob(id) {
  hideErrors();
  try {
    const job = await api(`/jobs/${id}`);
    state.job = job;
    saveLastJob();
    if (job.status === 'completed' && job.result) {
      renderResult(job.result);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      showEmptyResult();
      showError($('job-error'),
        `Задача ${STATUS_RU[job.status] || job.status}` + (job.error ? `: ${job.error}` : ''));
    } else {
      showEmptyResult();
      startPolling(id);
      $('upload-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    enableSubmit();
    refreshHistory();
  } catch (err) {
    if (err.status === 404) alert('Задача не найдена — возможно, она удалена по TTL.');
    else alert(err.detail || 'Не удалось открыть задачу');
  }
}

/* ========================= здоровье сервера ========================= */

async function refreshHealth() {
  const chip = $('server-status');
  try {
    const h = await api('/health');
    chip.className = 'status-chip online';
    const dev = h.device === 'cuda' ? 'GPU' : h.device === 'cpu' ? 'CPU' : (h.device || '?');
    chip.textContent = `API онлайн · ${dev}`;
    const note = $('ttl-note');
    note.textContent = h.job_ttl_seconds
      ? `результат хранится ≈ ${Math.round(h.job_ttl_seconds / 60)} мин после завершения`
      : '';
  } catch (e) {
    chip.className = 'status-chip offline';
    chip.textContent = 'API недоступен';
  }
}

/* ========================= localStorage: последняя задача ========================= */

function saveLastJob() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      id: state.job ? state.job.job_id : null,
      file: state.jobFileBase,
    }));
  } catch { /* приватный режим и т.п. */ }
}

function clearLastJob() {
  try { localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
}

function restoreLastJob() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LS_KEY) || 'null'); } catch { /* ignore */ }
  if (!saved || !saved.id) return;
  state.jobFileBase = saved.file || '';
  api(`/jobs/${saved.id}`)
    .then((job) => {
      state.job = job;
      if (job.status === 'completed' && job.result) {
        renderResult(job.result);
        clearLastJob();
      } else if (job.status === 'failed' || job.status === 'cancelled') {
        showError($('job-error'),
          `Задача ${STATUS_RU[job.status] || job.status}` + (job.error ? `: ${job.error}` : ''));
        clearLastJob();
      } else {
        startPolling(saved.id);
      }
    })
    .catch(() => clearLastJob());
}

/* ========================= инициализация ========================= */

function updatePromptNote() {
  const note = $('prompt-note');
  const current = ($('system-prompt').value || '').trim();
  const isDefault = current === DEFAULT_SYSTEM_PROMPT.trim();
  note.textContent = isDefault
    ? 'промт по умолчанию · суммаризация выполняется через локальный Ollama после завершения диаризации'
    : 'промт изменён и сохранён в этом браузере';
}

function initLlmPanel() {
  const promptEl = $('system-prompt'), modelEl = $('llm-model');
  try {
    promptEl.value = localStorage.getItem(LS_PROMPT_KEY) ?? DEFAULT_SYSTEM_PROMPT;
    modelEl.value = localStorage.getItem(LS_MODEL_KEY) || '';
  } catch (err) {
    promptEl.value = DEFAULT_SYSTEM_PROMPT;
    modelEl.value = '';
  }
  const persist = () => {
    try {
      localStorage.setItem(LS_PROMPT_KEY, promptEl.value);
      localStorage.setItem(LS_MODEL_KEY, modelEl.value.trim());
    } catch (err) { /* приватный режим браузера и т.п. */ }
    updatePromptNote();
  };
  promptEl.addEventListener('input', persist);
  modelEl.addEventListener('input', persist);
  $('prompt-reset').addEventListener('click', () => {
    promptEl.value = DEFAULT_SYSTEM_PROMPT;
    persist();
  });
  updatePromptNote();
  // Подсказка моделей из Ollama (GET /api/llm/models): backend сначала
  // берёт /api/ps (модели, уже загруженные в память), затем /api/tags
  // (установленные). Пустой список не оставляем молча — причина
  // из поля error показывается под полем модели.
  const note = $('llm-model-note');
  api('/llm/models')
    .then((data) => {
      const list = $('llm-model-list');
      if (!list || !Array.isArray(data.models)) return;
      list.innerHTML = '';
      for (const name of data.models) {
        const opt = document.createElement('option');
        opt.value = name;
        list.appendChild(opt);
      }
      if (note) {
        if (data.models.length) {
          note.textContent = data.source === 'ps'
            ? 'модели уже загружены в Ollama (api/ps)'
            : 'модели установлены в Ollama (api/tags)';
        } else {
          note.textContent = data.error
            ? 'Ollama: ' + data.error
            : 'в Ollama пока нет установленных моделей';
        }
      }
    })
    .catch(() => {
      if (note) note.textContent = 'Ollama: не удалось получить список моделей';
    });
}

async function onSummaryCopy() {
  const btn = $('summary-copy');
  // копируем исходный Markdown, а не отрендеренный текст
  const text = (state.summaryData && state.summaryData.summary) || $('summary-body').textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Скопировано ✓';
  } catch (err) {
    btn.textContent = 'Не удалось скопировать';
  }
  setTimeout(() => { btn.textContent = 'Копировать резюме'; }, 1500);
}

document.addEventListener('DOMContentLoaded', () => {
  const dz = $('dropzone'), input = $('file-input');

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('dragover'); }));
  dz.addEventListener('drop', (e) => setFile(e.dataTransfer.files && e.dataTransfer.files[0]));
  input.addEventListener('change', () => setFile(input.files && input.files[0]));

  $('file-clear').addEventListener('click', clearFile);

  $('num-speakers').addEventListener('input', () => syncSpeakerFields($('num-speakers')));
  $('min-speakers').addEventListener('input', () => syncSpeakerFields($('min-speakers')));
  $('max-speakers').addEventListener('input', () => syncSpeakerFields($('max-speakers')));

  $('submit-btn').addEventListener('click', onSubmit);
  $('cancel-btn').addEventListener('click', onCancel);

  for (const btn of document.querySelectorAll('.tab')) {
    btn.addEventListener('click', () => switchTab(btn.dataset.view));
  }

  $('copy-btn').addEventListener('click', onCopy);
  $('dl-txt').addEventListener('click', () => state.result &&
    download(`${exportBase()}.txt`, buildTxt(state.result), 'text/plain;charset=utf-8'));
  $('dl-srt').addEventListener('click', () => state.result &&
    download(`${exportBase()}.srt`, buildSrt(state.result), 'application/x-subrip;charset=utf-8'));
  $('dl-json').addEventListener('click', () => state.result &&
    download(`${exportBase()}.json`, JSON.stringify(state.result, null, 2), 'application/json'));

  $('history-refresh').addEventListener('click', refreshHistory);
  setInterval(refreshHistory, 15000);

  initLlmPanel();
  $('summary-rerun').addEventListener('click', () => requestSummary('manual'));
  $('summary-copy').addEventListener('click', onSummaryCopy);

  refreshHealth();
  setInterval(refreshHealth, 30000);
  refreshHistory();
  restoreLastJob();
});
