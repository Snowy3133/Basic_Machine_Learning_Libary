
#include "model.h"


typedef void (*mv_fn)(model_var* v);

static const mv_op_info mv_ops[MV_OP_COUNT] = {
    [MV_OP_CREATE]     = { "create",     0, NULL,           NULL           },
    [MV_OP_RELU]       = { "relu",       1, relu_fwd,       relu_bwd       },
    [MV_OP_ADD]        = { "add",        2, add_fwd,        add_bwd        },
    [MV_OP_SUB]        = { "sub",        2, sub_fwd,        sub_bwd        },
    [MV_OP_MATMUL]     = { "matmul",     2, matmul_fwd,     matmul_bwd     },
    [MV_OP_ADD_BIAS]   = { "add_bias",   2, add_bias_fwd,   add_bias_bwd   },
    [MV_OP_SOFTMAX_CE] = { "softmax_ce", 2, softmax_ce_fwd, softmax_ce_bwd },
};



model* model_create(memory_arena* arena, u32 max_vars);

model_var* mv_create(model* m, u32 rows, u32 cols, u32 flags);
static model_var* mv_new(model* m, mv_op op, u32 rows, u32 cols, u32 flags, model_var* a, model_var* b);

model_var* mv_relu(model* m, model_var* mv);
model_var* mv_add(model* m, model_var* a, model_var* b);
model_var* mv_sub(model* m, model_var* a, model_var* b);
model_var* mv_matmul(model* m, model_var* a, model_var* b);
model_var* mv_add_bias(model* m, model_var* x, model_var* bias);
