# LES Mackey-Glass 1400x Public Release

This package contains the public constrained Mackey-Glass demonstration only.

Included:
- `LES_MG_PUBLIC_CONSTRAINED_1000X_R1.py`
- `LES_MG_PUBLIC_CONSTRAINED_1000X_R1_RESULT.json`

Held-out result:
- Error-reduction ratio: `1440.6439924801339x`
- Baseline held-out NRMSE: `0.0014625428793411035`
- Candidate held-out NRMSE: `1.0152007622808114e-06`

Scope:
- Public demonstration.
- Train-only selection before held-out evaluation.
- Same historical series; not fresh data.
- This is intentionally not the private maximum-performance model.

Not included:
- Internal inverse-tuning tools
- Wall-location / wall-classification parameters
- Private search history
- Generic Digital-Matter machinery
- Private maximum-performance branches

Author:
Makoto Funakoshi / LES research
