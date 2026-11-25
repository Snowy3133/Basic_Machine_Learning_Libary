
#include <math.h>
#include "model.h"
#include "matrix.h"

typedef void (*mv_fn)(model_var* v);

static void relu_fwd(model_var* v) {
	mat_relu(v->val, v->inputs[0]->val);	

}

static void add_fwd(model_var* v) {
	mat_add(v->val, v->inputs[0]->val, v->inputs[1]->val);

}

static void sub_fwd(model_var* v) {
	mat_sub(v->val, v->inputs[0]->val, v->inputs[1]->val);

}

static void matmul_fwd(model_var* v) {
	mat_mul(v->val, v->inputs[0]->val, v->inputs[1]->val, true, false, false);

}

static void add_bias_fwd(model_var* v) {
	mat_add_bias(v->val, v->inputs[0]->val, v->inputs[1]->val);

}

static void softmax_ce_fwd(model_var* v) {
	mat_softmax_ce(v->val, v->aux, v->inputs[0]->val, v->inputs[1]->val);

}

static void relu_bwd(model_var* v) {
	model_var* a = v->inputs[0];

	if (a->grad != NULL) {
		mat_relu_add_grad(a->grad, a->val, v->grad);
	}

}

static void add_bwd(model_var* v) {
	model_var* a = v->inputs[0];
	model_var* b = v->inputs[1];

	if (a->grad != NULL) {
		mat_add(a->grad, a->grad, v->grad);
	}
	if (b->grad != NULL) {
		mat_add(b->grad, b->grad, v->grad);
	}
}

static void sub_bwd(model_var* v) {
	model_var* a = v->inputs[0];
	model_var* b = v->inputs[1];

	if (a->grad != NULL) {
		mat_add(a->grad, a->grad, v->grad);
	}
	if (b->grad != NULL) {
		mat_sub(b->grad, b->grad, v->grad);
	}
}

static void matmul_bwd(model_var* v) {
	model_var* a = v->inputs[0];
	model_var* b = v->inputs[1];

	if (a->grad != NULL) {
		mat_mul(a->grad, v->grad, b->val, false, false, true);
	}
	if (b->grad != NULL) {
		mat_mul(b->grad, a->val, v->grad, false, true, false);
	}
}

static void add_bias_bwd(model_var* v) {
	model_var* x = v->inputs[0];
	model_var* bias = v->inputs[1];

	if (x->grad != NULL) {
		mat_add(x->grad, x->grad, v->grad);
	}
	if (bias->grad != NULL) {
		mat_add_bias_grad(bias->grad, v->grad);
	}
}

static void softmax_ce_bwd(model_var* v) {
	model_var* logits = v->inputs[0];
	model_var* labels = v->inputs[1];

	if (logits->grad != NULL) {
		mat_softmax_ce_grad(logits->grad, v->aux, labels->val, v->grad);
	}
}

typedef struct {
	const char* name;
	u32 arity;
	mv_fn forward;
	mv_fn backward;
} mv_op_info;

static const mv_op_info mv_ops[MV_OP_COUNT] = {
    [MV_OP_CREATE]     = { "create",     0, NULL,            NULL           },
    [MV_OP_RELU]       = { "relu",       1, relu_fwd,        relu_bwd       },
    [MV_OP_ADD]        = { "add",        2, add_fwd,         add_bwd        },
    [MV_OP_SUB]        = { "sub",        2, sub_fwd,         sub_bwd        },
    [MV_OP_MATMUL]     = { "matmul",     2, matmul_fwd,      matmul_bwd     },
    [MV_OP_ADD_BIAS]   = { "add_bias",   2, add_bias_fwd,    add_bias_bwd   },
    [MV_OP_SOFTMAX_CE] = { "softmax_ce", 2, softmax_ce_fwd,  softmax_ce_bwd },
};



model* model_create(memory_arena* arena, u32 max_var) {
	model* mod = PUSH_STRUCT(arena, model);
	
	mod->arena = arena;
	mod->vars = PUSH_ARRAY(arena, model_var*, max_var);
	mod->num_vars = 0;
	mod->max_vars = max_var;

	return mod;
}

void model_forward(model* m) {
	for (u32 i = 0; i < m->num_vars; i++) {
		model_var* v = m->vars[i];

		if (v->op != MV_OP_CREATE) {
			mv_ops[v->op].forward(v);
		}
	}
}

void model_backward(model* m) {
	ASSERT(m->cost != NULL);

	for (u32 i = 0; i < m->num_vars; i++) {
		model_var* v = m->vars[i];
		if (v->grad != NULL && !(v->flags & MV_FLAG_PARAMETER)) {
			mat_clear(v->grad);
		}
	}

	mat_fill(m->cost->grad, 1.0f);

	for (u32 i = m->num_vars; i > 0; i--) {
		model_var* v = m->vars[i - 1];

		if (v->op != MV_OP_CREATE && v->grad != NULL) {
			mv_ops[v->op].backward(v);
		}
	}
}

static model_var* mv_new(model* m, mv_op op, u32 rows, u32 cols, u32 flags, model_var* a, model_var* b) {
	ASSERT(m->num_vars < m->max_vars);

	model_var* mv = PUSH_STRUCT(m->arena, model_var);

	mv->index = m->num_vars;
	if ((a != NULL && (a->flags & MV_FLAG_REQUIRES_GRAD)) ||
    	    (b != NULL && (b->flags & MV_FLAG_REQUIRES_GRAD))) {
		flags |= MV_FLAG_REQUIRES_GRAD;
	}
	mv->flags = flags;
	mv->op = op;
	mv->val = mat_create(m->arena, rows, cols);
	mv->inputs[0] = a;
	mv->inputs[1] = b;
	if (flags & MV_FLAG_REQUIRES_GRAD) {
		mv->grad = mat_create(m->arena, rows, cols);
	}

	m->vars[m->num_vars] = mv;
	m->num_vars++;

	return mv;

}

model_var* mv_create(model* m, u32 rows, u32 cols, u32 flags) {
	return mv_new(m, MV_OP_CREATE, rows, cols, flags, NULL, NULL);
}

model_var* mv_relu(model* m, model_var* mv) {
	return mv_new(m , MV_OP_RELU, mv->val->rows, mv->val->cols, 0, mv, NULL);
}

model_var* mv_add(model* m, model_var* a, model_var* b) {
	ASSERT(a->val->rows == b->val->rows && a->val->cols == b->val->cols);

	return mv_new(m, MV_OP_ADD, a->val->rows, b->val->cols, 0 , a ,b);
}

model_var* mv_sub(model* m, model_var* a, model_var* b) {
	ASSERT(a->val->rows == b->val->rows && a->val->cols == b->val->cols);
	
	return mv_new(m, MV_OP_SUB, a->val->rows, a->val->cols, 0, a, b);
}

model_var* mv_matmul(model* m, model_var* a, model_var* b) {
	ASSERT(a->val->cols == b->val->rows);

	return mv_new(m, MV_OP_MATMUL, a->val->rows, b->val->cols, 0, a , b);
}

model_var* mv_add_bias(model* m, model_var* x, model_var* bias) {
	ASSERT(x->val->cols == bias->val->cols && bias->val->rows == 1);

	return mv_new(m, MV_OP_ADD_BIAS, x->val->rows, x->val->cols, 0, x, bias);
}

model_var* mv_softmax_ce(model* m, model_var* logits, model_var* labels) {
	ASSERT(logits->val->rows == labels->val->rows && logits->val->cols == labels->val->cols);

	model_var* mv = mv_new(m, MV_OP_SOFTMAX_CE, 1, 1, 0, logits, labels);
	mv->aux = mat_create(m->arena, logits->val->rows, logits->val->cols);
	return mv;

}

model_var* mv_dense(model* m, model_var* x, u32 n_out) {
	u32 n_in = x->val->cols;
	u32 P = MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD;

	model_var* W = mv_create(m, n_in, n_out, P);
	model_var* b = mv_create(m, 1, n_out, P);

	f32 limit = sqrtf(6.0f / (f32)n_in);
	mat_fill_random(W->val, -limit, limit);

	return mv_add_bias(m, mv_matmul(m, x, W), b);
}

void mv_print(model* m) {
	printf("%4s  %-10s  %-10s  %-7s  %s\n", "node", "op", "shape", "inputs", "flags");

	for (u32 i = 0; i < m->num_vars; i++) {
		model_var* v = m->vars[i];

		char shape[24];
		snprintf(shape, sizeof(shape), "%u x %u", v->val->rows, v->val->cols);

		char inputs[24] = "-";
		if (v->inputs[0] != NULL && v->inputs[1] != NULL) {
			snprintf(inputs, sizeof(inputs), "%u, %u", v->inputs[0]->index, v->inputs[1]->index);
		} else if (v->inputs[0] != NULL) {
			snprintf(inputs, sizeof(inputs), "%u", v->inputs[0]->index);
		}

		char flags[8];
		u32 n = 0;
		if (v->flags & MV_FLAG_INPUT)          flags[n++] = 'I';
		if (v->flags & MV_FLAG_DESIRED_OUTPUT) flags[n++] = 'D';
		if (v->flags & MV_FLAG_PARAMETER)      flags[n++] = 'P';
		if (v->flags & MV_FLAG_REQUIRES_GRAD)  flags[n++] = 'G';
		if (v->flags & MV_FLAG_OUTPUT)         flags[n++] = 'O';
		if (v->flags & MV_FLAG_COST)           flags[n++] = 'C';
		flags[n] = '\0';

		printf("%4u  %-10s  %-10s  %-7s  %s\n", v->index, mv_ops[v->op].name, shape, inputs, flags);
	}

	printf("flags: I input, D desired output, P parameter, G needs a gradient, O output, C cost\n");
}



