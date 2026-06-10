import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import APIRouter, Request

from metrics import compute_all_metrics                
from frequentist import run_all_tests, srm_check        
from cuped import run_all_cuped                          
from bayesian import run_all_bayesian                    
from segmentation import run_all_segmentation            
from guardrails import run_all_guardrails                
from recommendation import make_recommendation           

from app.api.schemas import (
    SrmResponse, GuardrailsResponse, DecisionResponse,
)


router = APIRouter(tags=["results"])


@router.get("/topline")
def topline(request: Request) -> dict:
    """Descriptive metrics + SRM + frequentist inference (user-level)."""
    df = request.app.state.df
    return {
        "metrics": compute_all_metrics(df),
        "srm": srm_check(df),
        "frequentist": run_all_tests(df),
    }


@router.get("/srm", response_model=SrmResponse)
def srm(request: Request) -> dict:
    """Sample-ratio mismatch check (user-level chi-squared)."""
    return srm_check(request.app.state.df)


@router.get("/frequentist")
def frequentist(request: Request, analysis_unit: str = "user") -> dict:
    """Primary/secondary/guardrail tests. Pass analysis_unit=session to see naive view."""
    return run_all_tests(request.app.state.df, analysis_unit=analysis_unit)


@router.get("/cuped")
def cuped(request: Request) -> dict:
    """CUPED variance reduction — raw vs adjusted lifts per metric."""
    return run_all_cuped(request.app.state.df)


@router.get("/bayesian")
def bayesian(request: Request, analysis_unit: str = "user") -> dict:
    """Posterior P(T>C), credible intervals, expected loss per decision."""
    return run_all_bayesian(request.app.state.df, analysis_unit=analysis_unit)


@router.get("/segments")
def segments(request: Request) -> dict:
    """Heterogeneous treatment effects by device, country, source, medium."""
    return run_all_segmentation(request.app.state.df)


@router.get("/guardrails", response_model=GuardrailsResponse)
def guardrails(request: Request) -> dict:
    """Threshold-based pass/warning/fail per guardrail."""
    return run_all_guardrails(request.app.state.df)


@router.get("/decision", response_model=DecisionResponse)
def decision(request: Request) -> dict:
    """Integrated SHIP / HOLD / REJECT / INVESTIGATE call with rationale."""
    rec = make_recommendation(request.app.state.df)
    return {
        "verdict": rec["verdict"],
        "reasons": rec["reasons"],
        "summary": rec["summary"],
    }
