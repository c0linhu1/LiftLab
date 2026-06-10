from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    data_loaded: bool
    n_users: Optional[int] = None
    n_sessions: Optional[int] = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    experiment_name: str
    n_users: int
    n_sessions: int


class ExperimentDetail(BaseModel):
    experiment_id: str
    experiment_name: str
    hypothesis: str
    primary_metric: str
    guardrail_metrics: list[str]
    treatment_split: float
    stratify_by: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    seed: int
    n_users: int
    n_sessions: int
    ground_truth: dict


class SrmResponse(BaseModel):
    n_users: int
    n_treatment: int
    n_control: int
    expected_treatment_share: float
    actual_treatment_share: float
    chi2: float
    p_value: float
    passed: bool


class GuardrailRow(BaseModel):
    name: str
    column: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float
    direction: str
    fail_threshold: float
    warning_threshold: float
    status: str


class GuardrailsResponse(BaseModel):
    overall_status: str
    results: list[GuardrailRow]


class DecisionResponse(BaseModel):
    verdict: str
    reasons: list[str]
    summary: str
