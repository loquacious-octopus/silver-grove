# Stage 4: Dense Small Features

## Intent

Generate compact Three.js object reconstructions with dense small features,
repeated elements, real negative space, correct orientation, and attached
functional parts.

## Current Qwen3.5 Production Rules

- Use the prompt image and extracted ledger as the source of truth. Do not copy
  any reference-object identity into unrelated prompts.
- Build the primary body volume first. For shells, fans, dishes, bowls,
  medallions, panels, keys, and dense-feature objects, the body/silhouette must
  exist before rims, outlines, or decorative edges.
- Model holes, filigree, perforations, slots, rings, and openings with real
  empty space, transparent gaps, or separated geometry. Do not paint cutouts on
  solid faces.
- Repeated details need repeated small primitives with visible spacing/count:
  teeth, pins, slots, knurling, fan ribs, chain rollers, screws, ridges, and
  ornaments cannot be replaced by a smooth texture.
- Handles, grips, brackets, legs, rods, wheels, and supports must visibly
  penetrate, bracket, or overlap their parent part. Tangent contact often reads
  as disconnected in render grids.
- Preserve object identity over material polish. When a score is weak, repair
  in this order: body mass/silhouette, attachment, negative-space openings,
  repeated micro-features, then material/color.
- If the same flaw repeats, change construction tactic instead of restating the
  flaw. Examples: replace a solid bow with separated arcs, replace a rim tube
  with a shallow concave dish plus lip, or replace painted holes with separated
  panels around real gaps.
