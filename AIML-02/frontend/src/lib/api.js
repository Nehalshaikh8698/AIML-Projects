const API = '/api';

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === 'string'
      ? detail
      : detail?.message || JSON.stringify(detail || body);
    throw new Error(message || `Request failed (${response.status})`);
  }
  return body;
}

export const api = {
  health: () => request('/health'),
  studies: () => request('/studies'),
  study: (id) => request(`/studies/${id}`),
  audit: (id) => request(`/studies/${id}/audit`),
  upload: (formData) => request('/studies/upload', { method: 'POST', body: formData }),
  segmentCT: (formData) => request('/ct/segment', { method: 'POST', body: formData }),
  review: (id, payload) => request(`/studies/${id}/review`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }),
  orthancStudies: () => request('/orthanc/studies'),
  analyzeOrthanc: (id) => request(`/orthanc/studies/${id}/analyze`, { method: 'POST' }),
};

export function mediaUrl(path) {
  if (!path) return '';
  return path.startsWith('http') ? path : path;
}
