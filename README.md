# bevy-nui-plugins

Rust/Bevy integration plugins for Neon3 declarative UI.

The crate keeps Bevy gameplay ECS state in the host application while providing
the process boundary integration needed to connect Bevy with Neon3:

- `Neon3BevyPlugin` starts or connects to the Neon3 UI/WGPU services.
- `NeonScreenUi<V>` publishes fixed screen-space UI state.
- `NeonWorldUi<V>` publishes typed world anchors and instance rows.
- Camera frames and world anchor batches are paired by sequence.
- Pointer input is forwarded through the Neon3 protocol.
- Windows external surface interop imports the renderer-owned shared targets.

## Design Boundary

This crate does not own domain gameplay state. The Bevy application owns its
components and resources; Neon3 owns declarative layout, text, hit testing,
presentation state, and final UI pixels. The crate communicates through the
typed Neon3 IPC/protocol contracts rather than exposing UI element IDs as game
commands.

## Example

```toml
[dependencies]
bevy = "0.19.1"
bevy-nui-plugins = "0.1"
```

See the `bevy-nui-host` example repository for a complete physics playground
and combined screen/world UI case.

World billboard panels keep their authored screen size while retaining their
real camera distance for depth-tested scene occlusion.
