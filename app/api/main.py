import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clean import build_clean_sessions                      
from assign_experiment import assign_users, DEFAULT_EXPERIMENT 
from simulate_treatment import (                          
    simulate_treatment_effects, get_ground_truth,
)

from app.api.routes_experiments import router as experiments_router  
from app.api.routes_results import router as results_router         
from app.api.schemas import HealthResponse                      


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot-time pipeline: load, assign, simulate, cache."""
    print("[startup] building clean sessions...")
    df = build_clean_sessions()
    print(f"[startup]   {len(df):,} clean sessions loaded.")
    print("[startup] assigning users to variants...")
    df = assign_users(df)
    print("[startup] simulating treatment effects...")
    df = simulate_treatment_effects(df)
    print(
        f"[startup] cached: {len(df):,} sessions, "
        f"{df['user_pseudo_id'].nunique():,} users."
    )

    app.state.df = df
    app.state.config = DEFAULT_EXPERIMENT
    app.state.ground_truth = get_ground_truth()
    yield



app = FastAPI(
    title="LiftLab",
    version="0.2.0",
    description="A/B testing platform — experimentation, inference, decisioning.",
    lifespan=lifespan,
)

app.include_router(experiments_router)
app.include_router(results_router)


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    state = request.app.state
    has_data = hasattr(state, "df")
    return HealthResponse(
        status="ok",
        service="liftlab-api",
        data_loaded=has_data,
        n_users=int(state.df["user_pseudo_id"].nunique()) if has_data else None,
        n_sessions=int(len(state.df)) if has_data else None,
    )
