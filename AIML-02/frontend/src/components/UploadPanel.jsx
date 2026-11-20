import React, { useRef, useState } from 'react';
import { FileImage, UploadCloud, X } from 'lucide-react';

export default function UploadPanel({ onUpload, busy }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [patientRef, setPatientRef] = useState('STUDY-' + new Date().toISOString().slice(0, 10).replaceAll('-', ''));
  const [notes, setNotes] = useState('');

  const submit = async (event) => {
    event.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('patient_ref', patientRef);
    form.append('notes', notes);
    await onUpload(form);
    setFile(null);
    setNotes('');
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <form className="upload-panel" onSubmit={submit}>
      <div className="panel-heading">
        <div><span className="eyebrow">NEW ANALYSIS</span><h2>Upload an image</h2></div>
        <span className="icon-chip"><UploadCloud size={18} /></span>
      </div>
      <button type="button" className={`dropzone ${file ? 'has-file' : ''}`} onClick={() => inputRef.current?.click()}>
        <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom,image/png,image/jpeg,application/dicom" hidden
          onChange={(event) => setFile(event.target.files?.[0] || null)} />
        {file ? <><FileImage size={25} /><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB · Ready to analyze</small></>
          : <><span className="upload-icon"><UploadCloud size={24} /></span><strong>Choose a scan to upload</strong><small>PNG, JPG or DICOM · Max 20 MB</small></>}
      </button>
      {file && <button type="button" className="clear-file" onClick={() => { setFile(null); inputRef.current.value = ''; }}><X size={14} /> Remove file</button>}
      <label className="field-label">Study reference <span>Use a non-identifying code</span></label>
      <input className="text-input" value={patientRef} onChange={(e) => setPatientRef(e.target.value)} maxLength={80} placeholder="e.g. STUDY-0001" />
      <label className="field-label">Notes <span>Optional</span></label>
      <textarea className="text-input notes-input" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={1000} placeholder="Add context for the reviewer..." />
      <button className="primary-button full-button" type="submit" disabled={!file || busy}>
        {busy ? <><span className="spinner" /> Analyzing image...</> : <>Run AI analysis <span>→</span></>}
      </button>
      <p className="privacy-note">Do not upload identifiable patient information. This demo stores image previews locally.</p>
    </form>
  );
}
