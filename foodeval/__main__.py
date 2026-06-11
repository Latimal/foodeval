"""Allow running FoodEval as ``python -m foodeval``."""

import sys

from foodeval.cli import main

sys.exit(main())
