---
name: submit
description: Package and finalize the submission file per docs/io_rules.md. Only ever run explicitly via /submit — never auto-triggered by the model.
disable-model-invocation: true
---

# Submit

Package the final resolved output into the exact submission format required
by the competition, and finalize it.

## Steps
1. Confirm `score` has been run and reviewed on the current output — do not
   submit unscored output.
2. Re-read `docs/io_rules.md` and validate the final output against it
   exactly: file name, location, columns, dtypes, ordering, encoding.
3. Copy/write the final submission file to the required location.
4. Report the final file path and a summary of what was submitted (score,
   row count, any last-minute caveats).

## Hard rule
This skill must never run automatically as part of another skill's flow.
It only runs when the user explicitly invokes `/submit`.
