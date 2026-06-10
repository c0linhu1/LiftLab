import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ExperimentSummary, ExperimentDetail


router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentSummary])
def list_experiments(request: Request):
    config = request.app.state.config
    df = request.app.state.df
    return [
        ExperimentSummary(
            experiment_id=config.experiment_id,
            experiment_name=config.experiment_name,
            n_users=int(df["user_pseudo_id"].nunique()),
            n_sessions=int(len(df)),
        )
    ]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str, request: Request):
    config = request.app.state.config
    df = request.app.state.df
    if experiment_id != config.experiment_id:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")

    return ExperimentDetail(
        experiment_id=config.experiment_id,
        experiment_name=config.experiment_name,
        hypothesis=config.hypothesis,
        primary_metric=config.primary_metric,
        guardrail_metrics=config.guardrail_metrics,
        treatment_split=config.treatment_split,
        stratify_by=config.stratify_by,
        start_date=config.start_date,
        end_date=config.end_date,
        seed=config.seed,
        n_users=int(df["user_pseudo_id"].nunique()),
        n_sessions=int(len(df)),
        ground_truth=request.app.state.ground_truth,
    )
