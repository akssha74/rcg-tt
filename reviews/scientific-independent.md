# Independent scientific default-reject review

`reviewer_kind=agent`  
`authoring_pass=false`  
`agent_id=77186efa-3a1c-495a-98b5-7d5a570ce814`

## Presumption

NOT-GREAT.

## Strongest rejection at review time

The frozen corrective preregistration defined W13 on means across seeds, but
the implementation silently applied an all-seed rule and reported W13 as
failed. The review also found that summary JSON did not include portable
per-example scores, that the archive retained absolute paths, and that the
reference-dependent/received score difference changed both information access
and degradation regime.

## Standing findings at review time

1. Correct W13 to pass under the registered rule and label all-seed robustness
   post hoc.
2. Release per-example scores, predictions, transitions and site/cluster IDs.
3. Normalize paths, pin dependencies, remove parent-tree generator
   dependencies and demonstrate a clean end-to-end reproduction.
4. Add acquisition-selection counts, satellite-overlap dependence analysis and
   site-specific sensitivity.
5. Treat the main quantity as descriptive rather than a causal
   hidden-information effect.
6. Discuss TARDIS/EarthShift and the limits of source-only baseline comparison.
7. Broaden architecture/event evidence or narrow the contribution.

## Verdict

NOT-GREAT.
