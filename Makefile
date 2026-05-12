ARGS ?=

# Capture any extra positional goals so they can be passed through to the command.
# Example: `make fid-score cifar_real fid_samples --device cuda --batch-size 50`
EXTRA_ARGS := $(filter-out $@,$(MAKECMDGOALS))

.PHONY: fid-score

fid-score:
	uv run python -m pytorch_fid $(ARGS) $(EXTRA_ARGS)

# Swallow extra goals so make doesn't try to build them as targets.
%:
	@:
