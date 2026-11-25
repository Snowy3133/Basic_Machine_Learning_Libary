

#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"
#include "arena.h"

typedef enum {
	MV_FLAG_REQUIRES_GRAD = 1 << 0,
	MV_FLAG_PARAMETER = 1 << 1,
	MV_FLAG_INPUT = 1 << 2,
	MV_FLAG_OUTPUT = 1 << 3,
	MV_FLAG_DESIRED_OUTPUT = 1 << 4,
	MV_FLAG_COST = 1 << 5,
} mv_flags;


typedef enum {
	MV_OP_CREATE,
	MV_OP_RELU,
	MV_OP_ADD,
	MV_OP_SUB,
	MV_OP_MATMUL,
	MV_OP_ADD_BIAS,
	MV_OP_SOFTMAX_CE,
	MV_OP_COUNT,
} mv_op;


typedef struct model_var {
	u32 index;
	u32 flags;
	mv_op op;
	struct model_var* inputs[2];
	matrix* val;
	matrix* grad;
	matrix* aux;
} model_var;


typedef struct {
	memory_arena* arena;
	model_var** vars;
	u32 num_vars;
	u32 max_vars;
	model_var* input;
	model_var* desired_output;
	model_var* output;
	model_var* cost;
} model;

model* model_create(memory_arena* arena, u32 max_vars);
model_var* mv_create(model* m, u32 rows, u32 cols, u32 flags);
model_var* mv_relu(model* m, model_var* mv);
model_var* mv_add(model* m, model_var* a, model_var* b);
model_var* mv_sub(model* m, model_var* a, model_var* b);
model_var* mv_matmul(model* m, model_var* a, model_var* b);
model_var* mv_add_bias(model* m, model_var* x, model_var* bias);
model_var* mv_softmax_ce(model* m, model_var* logits, model_var* labels);
model_var* mv_dense(model* m, model_var* x, u32 n_out);

void mv_print(model* m);
#endif

