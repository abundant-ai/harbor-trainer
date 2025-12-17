from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress verbose logs
    logging.getLogger("harbor").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def run_training(config_path: Path):
    """
    Launch Prime-RL training with the given config.
    
    This wraps prime-rl's `rl` command for convenience.
    """
    # Ensure prime-rl directory exists
    prime_rl_dir = Path(__file__).parent.parent / "prime-rl"
    if not prime_rl_dir.exists():
        logger.error(f"prime-rl directory not found at {prime_rl_dir}")
        sys.exit(1)
    
    # Build command
    cmd = [
        "uv", "run",
        "--directory", str(prime_rl_dir),
        "rl", "@", str(config_path.absolute()),
    ]
    
    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Set PYTHONPATH to include src directory so prime-rl can import harbor_env
    env = os.environ.copy()
    src_dir = Path(__file__).parent.parent
    python_path = env.get("PYTHONPATH", "")
    if python_path:
        env["PYTHONPATH"] = f"{src_dir}:{python_path}"
    else:
        env["PYTHONPATH"] = str(src_dir)
    
    # Run training
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(130)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Harbor RL training with Prime-RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train 8B model with LoRA on Harbor tasks:
    python -m src.train --config configs/harbor_8b.toml
    
    # Train with Modal cloud sandboxes:
    python -m src.train --config configs/harbor_8b_modal.toml
    
    # Direct prime-rl usage (alternative):
    cd harbortrainer
    PYTHONPATH=. uv run --directory prime-rl rl @ configs/harbor_8b.toml
""",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to prime-rl config file (TOML)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    run_training(args.config)


if __name__ == "__main__":
    main()
