import React, { useRef, useState } from 'react';
import { Layers3, UploadCloud } from 'lucide-react';

export default function CTUploadPanel({ onSegment, busy, ready }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [patientRef, setPatientRef] = useState('CT-STUDY-' + new Date().toISOString().slice(0, 10).replaceAll('-', ''));

  const submit = async (event) => {
    event.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('patient_ref', patientRef);
    await onSegment(form);
    setFile(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <form className="upload-panel ct-upload-panel" onSubmit={submit}>
      <div className="panel-heading">
        <div><span className="eyebrow">3D CT WORKFLOW</span><h2>Segment CT volume</h2></div>
        <span className="icon-chip violet-chip"><Layers3 size={18} /></span>
      </div>
      <button type="button" className={`dropzone ct-dropzone ${file ? 'has-file' : ''}`} onClick={() => inputRef.current?.click()}>
        <input ref={inputRef} type="file" accept=".nii,.nii.gz" hidden onChange={(event) => setFile(event.target.files?.[0] || null)} />
        <span className="upload-icon"><UploadCloud size={23} /></span>
        <strong>{file ? file.name : 'Choose a CT NIfTI volume'}</strong>
        <small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ready` : 'NIfTI .nii / .nii.gz · Max 200 MB'}</small>
      </button>
      <label className="field-label">Study reference <span>Non-identifying code</span></label>
      <input className="text-input" value={patientRef} onChange={(e) => setPatientRef(e.target.value)} maxLength={80} />
      <button className="primary-button full-button" type="submit" disabled={!file || busy || !ready}>
        {busy ? <><span className="spinner" /> Segmenting volume...</> : <>Run 3D segmentation <span>→</span></>}
      </button>
      <p className="privacy-note">Requires a locally trained MONAI 3D U-Net checkpoint. Only a center-slice preview is displayed.</p>
      {!ready && <p className="ct-model-status">3D model not trained yet · see setup instructions below.</p>}
    </form>
  );
}
