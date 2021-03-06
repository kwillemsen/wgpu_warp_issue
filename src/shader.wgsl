//
// Vertex shader
//
struct VertexInput {
    [[location(0)]] position: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}



//
// Fragment shader
//
[[block]]
struct ColorUniform {
    color: vec4<f32>;
};
[[group(0), binding(0)]]
var<uniform> color: ColorUniform;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return color.color;
}
