# Future Integration: tile-compiler

## Current State
Compiles game strategies into fast lookup tables via tile-based field training. Zero dependencies (core), optional GPU via torch. Five-pass optimization pipeline: train → compile → optimize → evaluate → deploy. Strategies are encoded as tiles (state fragments) that map to actions.

## Integration Opportunities

### With ESP32 room compilation
tile-compiler targets ESP32 by compiling strategies into lookup tables that fit in 520KB SRAM. A room running on ESP32 uses compiled tiles for instant decision-making: sensor pattern → tile lookup → action. No ML, no heap, no computation — just table lookup.

### With construct-core Layer 0
Compiled tiles ARE construct-core's Layer 0 lookup tables. tile-compiler produces the tables; Layer 0's `query_lookup()` reads them. The compilation pipeline descends through construct-core's tiers: Layer 2 trains the strategy (full compute), Layer 1 compiles it (sync), Layer 0 deploys it (bare metal).

### With lever-runner
tile-compiler's "teach once, run forever" model mirrors lever-runner's trust compilation. lever-runner compiles shell commands; tile-compiler compiles game strategies. Both produce lookup tables for instant execution. The unified framework: any domain-specific behavior can be compiled into a fast lookup table.

## Dormant Ideas Now Unlockable
The compilation pipeline had no deployment target. Now construct-core's three tiers provide the target, and ESP32 rooms provide the extreme-constraint test case. The compiler can now produce real deployed code, not just research outputs.

## Potential in Mature Systems
Every room's strategies are compiled via tile-compiler. Hot paths become lookup tables. The room gets faster over time as more strategies are compiled. Eventually, most room decisions are O(1) table lookups — the room's "muscle memory."

## Cross-Pollination Ideas
- **tile-cuda/tile-neon/tile-opencl**: Hardware-specific compilation targets
- **agentic-compiler**: General compilation framework; tile-compiler is the strategy specialization
- **lever-runner-carapace**: Shared BLAKE2b hashing for tile state identification

## Dependencies for Next Steps
- ESP32 target compilation (table generation for 520KB SRAM)
- Integration with construct-core Layer 0 query_lookup
- Room-specific strategy compilation pipeline
