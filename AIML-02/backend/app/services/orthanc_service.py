from typing import Any

import requests


class OrthancClient:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)

    def _get(self, path: str):
        response = requests.get(f"{self.base_url}{path}", auth=self.auth, timeout=15)
        response.raise_for_status()
        return response

    def list_studies(self) -> list[dict[str, Any]]:
        identifiers = self._get("/studies").json()
        studies = []
        for study_id in identifiers[:100]:
            try:
                item = self._get(f"/studies/{study_id}").json()
                main = item.get("MainDicomTags", {})
                patient = item.get("PatientMainDicomTags", {})
                studies.append({
                    "orthanc_id": study_id,
                    "study_date": main.get("StudyDate"),
                    "description": main.get("StudyDescription"),
                    "patient_id": patient.get("PatientID"),
                    "instances": len(item.get("Instances", [])),
                })
            except requests.RequestException:
                continue
        return studies

    def first_instance(self, study_id: str) -> tuple[bytes, str]:
        study = self._get(f"/studies/{study_id}").json()
        instance_ids = study.get("Instances", [])
        if not instance_ids:
            raise ValueError("This Orthanc study contains no instances.")
        content = self._get(f"/instances/{instance_ids[0]}/file").content
        return content, f"{instance_ids[0]}.dcm"
