import React from 'react';

export default function StatusPill({ status }) {
  const label = status === 'approved' ? 'Reviewed · Approved'
    : status === 'rejected' ? 'Reviewed · Rejected' : 'Awaiting review';
  return <span className={`status-pill ${status || 'pending'}`}><i />{label}</span>;
}
