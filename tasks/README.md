# Skill-OPD Task Tracker

This directory tracks project work that should survive across local and server
development machines.

Current source of truth:

- GitHub repository: `thinkingmanyangyang/verl-agent-opd`
- Main branch: `main`
- Active code root: this `verl-agent-opd` repository
- Reference-only local cache: outside this repo, under `resource/`

Workflow:

1. Pull before starting work.
2. Keep code, scripts, and task updates in small commits.
3. Use the server for GPU rollout/training runs.
4. Do not commit generated rollouts, checkpoints, logs, model weights, or caches.
5. Commit useful findings, run commands, and next steps back into `tasks/` or
   `docs/skill_opd/`.

Task files:

- `roadmap.md`: full project roadmap from rollout export to Skill-OPD training.
- `phase1_sokoban_rollout.md`: server checklist for the first offline rollout.
- `backlog.md`: concrete short-term tasks and blockers.

