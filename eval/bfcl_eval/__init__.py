"""
Local copy of BFCL evaluation package (moved under eval/).
Alias top-level name 'bfcl_eval' to maintain internal absolute imports.
"""

import sys

# Ensure internal absolute imports (bfcl_eval.*) continue to work
sys.modules.setdefault("bfcl_eval", sys.modules[__name__])
