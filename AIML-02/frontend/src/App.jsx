import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowUpRight, CheckCircle2, ChevronRight,
  Database, FilePlus2, HeartPulse, RefreshCw, Server, ShieldAlert,
} from 'lucide-react';
import { api } from './lib/api';
import UploadPanel from './components/UploadPanel';
import CTUploadPanel from './components/CTUploadPanel';
import StudyViewer from './components/StudyViewer';
import StatusPill from './components/StatusPill';

function App() {
  const [health, setHealth] = useState(null);
  const [studies, setStudies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [audit, setAudit] = useState([]);
  const [busy, setBusy] = useState(false);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [orthancOpen, setOrthancOpen] = useState(false);
  const [orthancStudies, setOrthancStudies] = useState([]);
  const [orthancBusy, setOrthancBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const refresh = useCallback(async (selectedId = selected?.id) => {
    const [healthResult, studyResult] = await Promise.all([api.health(), api.studies()]);
    setHealth(healthResult);
    setStudies(studyResult);
    if (selectedId) {
      const current = studyResult.find((item) => item.id === selectedId);
      if (current) {
        const [full, events] = await Promise.all([api.study(selectedId), api.audit(selectedId)]);
        setSelected(full);
        setAudit(events);
      }
    }
  }, [selected?.id]);

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  const selectStudy = async (id) => {
    try {
      setError('');
      const [full, events] = await Promise.all([api.study(id), api.audit(id)]);
      setSelected(full);
      setAudit(events);
    } catch (err) { setError(err.message); }
  };

  const handleUpload = async (formData) => {
    try {
      setBusy(true); setError(''); setNotice('');
      const result = await api.upload(formData);
      await refresh(result.id);
      setNotice('Image analyzed and saved to the review queue.');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  const handleCTSegment = async (formData) => {
    try {
      setBusy(true); setError(''); setNotice('');
      const result = await api.segmentCT(formData);
      await refresh(result.id);
      setNotice('CT volume segmented and added to the review queue.');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  const handleReview = async (decision, reviewer, comments) => {
    if (!selected) return;
    try {
      setReviewBusy(true); setError('');
      await api.review(selected.id, { decision, reviewer, comments });
      await refresh(selected.id);
      setNotice(`Review decision recorded: ${decision}.`);
    } catch (err) { setError(err.message); }
    finally { setReviewBusy(false); }
  };

  const loadOrthanc = async () => {
    try {
      setOrthancBusy(true); setError('');
      const items = await api.orthancStudies();
      setOrthancStudies(items); setOrthancOpen(true);
    } catch (err) { setError(err.message); }
    finally { setOrthancBusy(false); }
  };

  const analyzeOrthanc = async (id) => {
    try {
      setOrthancBusy(true); setError('');
      const result = await api.analyzeOrthanc(id);
      await refresh(result.id);
      setOrthancOpen(false);
      setNotice('Orthanc study imported and analyzed.');
    } catch (err) { setError(err.message); }
    finally { setOrthancBusy(false); }
  };

  const pendingCount = studies.filter((item) => item.review_status === 'pending').length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><HeartPulse size={22} /></div><div><strong>medscan<span>AI</span></strong><small>IMAGING REVIEW</small></div></div>
        <div className="sidebar-label">WORKSPACE</div>
        <button className="nav-item active"><Activity size={17} /> Review dashboard <span className="nav-count">{pendingCount}</span></button>
        <button className="nav-item" onClick={loadOrthanc}><Database size={17} /> DICOM studies <ArrowUpRight size={14} className="nav-arrow" /></button>
        <div className="sidebar-spacer" />
        <div className="sidebar-note"><ShieldAlert size={16} /><div><strong>Research build</strong><p>Not approved for clinical use. No patient care decisions.</p></div></div>
        <div className="sidebar-footer"><span className="online-dot" /> Local environment <span>v1.0</span></div>
      </aside>

      <main className="main-area">
        <header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={14} /> <strong>Review dashboard</strong></div><div className="topbar-right"><span className="local-badge"><span /> LOCAL INSTANCE</span><button className="icon-button" title="Refresh" onClick={() => refresh().catch((err) => setError(err.message))}><RefreshCw size={16} /></button></div></header>

        <div className="content-wrap">
          <section className="page-heading"><div><div className="eyebrow">MEDICAL IMAGING · RESEARCH WORKFLOW</div><h1>Imaging review workspace</h1><p>Analyze de-identified chest X-rays, inspect model visualizations, and record reviewer decisions.</p></div><div className="date-stamp"><span>TODAY</span><strong>{new Date().toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })}</strong></div></section>

          <section className="notice-banner"><div className="notice-icon"><AlertTriangle size={17} /></div><div><strong>Educational research prototype — not for clinical use</strong><p>This is an educational workflow only. Demo-mode scores are simulated; trained-model outputs and Grad-CAM are experimental. Nothing here is a diagnosis or for patient care.</p></div></section>

          {error && <div className="alert error-alert"><AlertTriangle size={17} /><span>{error}</span><button onClick={() => setError('')}>×</button></div>}
          {notice && <div className="alert success-alert"><CheckCircle2 size={17} /><span>{notice}</span><button onClick={() => setNotice('')}>×</button></div>}

          <section className="metric-row">
            <div className="metric-card"><span className="metric-icon blue"><FilePlus2 size={18} /></span><div><small>Studies in workspace</small><strong>{studies.length}</strong></div></div>
            <div className="metric-card"><span className="metric-icon amber"><Activity size={18} /></span><div><small>Awaiting review</small><strong>{pendingCount}</strong></div></div>
            <div className="metric-card"><span className="metric-icon green"><CheckCircle2 size={18} /></span><div><small>Reviewed</small><strong>{studies.filter((item) => item.review_status !== 'pending').length}</strong></div></div>
            <div className="metric-card system-card"><span className="metric-icon violet"><Server size={18} /></span><div><small>Model status</small><strong className={health?.model_loaded ? 'ready-text' : 'waiting-text'}>{health?.model_loaded ? 'Ready' : health?.demo_mode_enabled ? 'Demo mode' : 'Not trained'}</strong></div></div>
          </section>

          {!health?.model_loaded && <div className="model-warning"><div><AlertTriangle size={18} /><strong>{health?.demo_mode_enabled ? 'Demo mode is active — not AI inference' : 'Model checkpoint not found'}</strong></div><p>{health?.demo_mode_enabled ? 'Uploads can demonstrate the review workflow using simulated image-statistics scores and an illustrative map. Train the DenseNet121 model before describing results as AI predictions.' : (health?.model_message || 'Train the model before uploading scans.')}</p><code>python ml/train.py --data-dir data/NIH-ChestXray14/images --labels-csv data/NIH-ChestXray14/Data_Entry_2017_v2020.csv</code></div>}
          {!health?.segmentation_model_loaded && <div className="model-warning segmentation-warning"><div><AlertTriangle size={18} /><strong>3D segmentation model not trained</strong></div><p>{health?.segmentation_model_message || 'Download the MSD Task06_Lung dataset and train the MONAI model.'}</p><code>python ml/train_segmentation.py --data-dir data/Task06_Lung --epochs 100</code></div>}

          <div className="workspace-grid">
            <div className="left-column">
              <UploadPanel onUpload={handleUpload} busy={busy} />
              <CTUploadPanel onSegment={handleCTSegment} busy={busy} ready={Boolean(health?.segmentation_model_loaded)} />
              <section className="card studies-card">
                <div className="section-title-row"><div><span className="eyebrow">RECENT ACTIVITY</span><h2>Study queue</h2></div><span className="count-badge">{studies.length} total</span></div>
                {studies.length ? <div className="study-list">{studies.map((study) => (
                  <button key={study.id} className={`study-item ${selected?.id === study.id ? 'selected' : ''}`} onClick={() => selectStudy(study.id)}>
                    <span className="study-thumb"><img src={study.image_url} alt="" /></span>
                    <span className="study-info"><strong>{study.patient_ref}</strong><small>{new Date(study.created_at).toLocaleDateString()} · {study.source || 'upload'}</small></span>
                    <StatusPill status={study.review_status} />
                  </button>
                ))}</div> : <div className="queue-empty"><FilePlus2 size={22} /><p>No studies yet</p><small>New uploads will appear here.</small></div>}
              </section>
            </div>
            <div className="right-column"><StudyViewer study={selected} onReview={handleReview} reviewBusy={reviewBusy} audit={audit} /></div>
          </div>

          <footer className="app-footer"><span>MedScan AI · Academic prototype</span><span>Data storage: {health?.database || 'checking'} · Model: DenseNet121 / NIH-14</span><a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API documentation ↗</a></footer>
        </div>
      </main>

      {orthancOpen && <div className="modal-backdrop" onClick={() => setOrthancOpen(false)}><div className="orthanc-modal" onClick={(event) => event.stopPropagation()}>
        <div className="section-title-row"><div><span className="eyebrow">DICOM SERVER</span><h2>Studies in Orthanc</h2></div><button className="icon-button" onClick={() => setOrthancOpen(false)}>×</button></div>
        <p className="modal-description">Select a de-identified study to import its first DICOM instance for this MVP.</p>
        {orthancStudies.length ? <div className="orthanc-list">{orthancStudies.map((item) => <div className="orthanc-row" key={item.orthanc_id}><div><strong>{item.description || 'Chest imaging study'}</strong><small>{item.study_date || 'Date unavailable'} · {item.instances} instance(s)</small></div><button className="small-primary" disabled={orthancBusy || !health?.model_loaded} onClick={() => analyzeOrthanc(item.orthanc_id)}>{orthancBusy ? 'Working...' : 'Analyze'}</button></div>)}</div> : <div className="queue-empty"><Database size={22} /><p>No Orthanc studies found</p><small>Upload a de-identified DICOM study to Orthanc Explorer first.</small></div>}
        <div className="modal-footer"><span>Orthanc local service · Port 8042</span><button className="text-button" onClick={loadOrthanc}><RefreshCw size={14} /> Refresh list</button></div>
      </div></div>}
    </div>
  );
}

export default App;
