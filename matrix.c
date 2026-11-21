
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "arena.h"

matrix* mat_create(memory_arena* arena, u32 rows, u32 cols) {
	matrix* mat = PUSH_STRUCT(arena, matrix);

	if (mat == NULL) {
		return NULL;
	}

	mat->rows = rows;
	mat->cols = cols;
	
	mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
	
	if (mat->data == NULL) {
		return NULL;
	}

	return mat;

}

matrix* mat_load(memory_arena* arena, u32 rows, u32 cols, const char* filename);

b32 mat_copy(matrix* dst, matrix* src) {
	if (dst->rows != src-> rows || dst->cols != src->cols) {
		return false;
	}

	memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

	return true;

}

void mat_clear(matrix* mat) {
	memset(mat->data, 0, sizeof(f32) * mat->rows * mat->cols); 

}

void mat_fill(matrix* mat, f32 fill) {
	u64 size = (u64) mat->cols * mat->rows;

	for (u64 i = 0; i < size; i++) {
		mat->data[i] = fill;
	}
}

void mat_fill_random(matrix* mat, f32 lower, f32 upper) {
	u64 size = (u64) mat->cols * mat->rows;

	for (u64 i = 0; i < size; i++) {
		mat->data[i] = lower + (upper - lower) * ((f32)rand() / RAND_MAX);
	}
}

void mat_scale(matrix* mat, f32 scale) {
	u64 size = (u64) mat->cols * mat->rows;

	for (u64 i = 0; i < size; i++) {
		mat->data[i] *= scale;
	}
}

f32 mat_sum(matrix* mat) {
	u64 size = (u64) mat->cols * mat->rows;

	f32 sum = 0.0f;
	for (u64 i = 0; i < size; i++) {
		sum += mat->data[i];
	}

	return sum;

}

u64 mat_argmax(matrix* mat) {
	u64 size = (u64) mat->cols * mat->rows;

	u64 max_i = 0;

	for (u64 i = 0; i < size; i++) {
		if (mat->data[i] > mat->data[max_i]) {
			max_i = i;
		}
	}

	return max_i;

}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
	if (a->rows != b->rows || a->cols != b->cols) {
		return false;
	}

	if (a->rows != out->rows || a->cols != out->cols) {
		return false;
	}

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = a->data[i] + b->data[i];
	}

	return true;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
	if (a->rows != b->rows || a->cols != b->cols) {
		return false;
	}

	if (a->rows != out->rows || a->cols != out->cols) {
		return false;
	}

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = a->data[i] - b->data[i];
	}

	return true;

}

void _mat_mul_nn(matrix* out, matrix* a, matrix* b) {
    for (u64 out_row = 0; out_row < out->rows; out_row++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 out_col = 0; out_col < out->cols; out_col++) {
                out->data[out_col + out_row * out->cols] +=
                    a->data[k + out_row * a->cols] *
                    b->data[out_col + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, matrix* a, matrix* b) {
    for (u64 out_row = 0; out_row < out->rows; out_row++) {
        for (u64 out_col = 0; out_col < out->cols; out_col++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[out_col + out_row * out->cols] +=
                    a->data[k + out_row * a->cols] *
                    b->data[k + out_col * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, matrix* a, matrix* b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 out_row = 0; out_row < out->rows; out_row++) {
            for (u64 out_col = 0; out_col < out->cols; out_col++) {
                out->data[out_col + out_row * out->cols] +=
                    a->data[out_row + k * a->cols] *
                    b->data[out_col + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, matrix* a, matrix* b) {
    for (u64 out_row = 0; out_row < out->rows; out_row++) {
        for (u64 out_col = 0; out_col < out->cols; out_col++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[out_col + out_row * out->cols] +=
                    a->data[out_row + k * a->cols] *
                    b->data[k + out_col * b->cols];
            }
        }
    }
}

b32 mat_mul(matrix* out, matrix* a, matrix* b, b8 zero_out, b8 transpose_a, b8 transpose_b) {
    u64 a_rows = transpose_a ? a->cols : a->rows;
    u64 a_cols = transpose_a ? a->rows : a->cols;
    u64 b_rows = transpose_b ? b->cols : b->rows;
    u64 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) return false;
    if (out->rows != a_rows || out->cols != b_cols) return false;
    if (out->data == a->data || out->data == b->data) return false;

    if (zero_out) mat_fill(out, 0.0f);

    if      (!transpose_a && !transpose_b) _mat_mul_nn(out, a, b);
    else if (!transpose_a &&  transpose_b) _mat_mul_nt(out, a, b);
    else if ( transpose_a && !transpose_b) _mat_mul_tn(out, a, b);
    else                                   _mat_mul_tt(out, a, b);

    return true;
}

b32 mat_relu(matrix* out, const matrix* in) {
	if (in->rows != out->rows || in->cols != out->cols) {
		return false;
	}
	
	u64 size = (u64) out->rows * out->cols;
	for (u64 i = 0; i < size; i++) {
		out->data[i] = MAX(0, in->data[i]);
	}

	return true;

}

b32 mat_softmax(matrix* out, const matrix* in) {	
	if (in->rows != out->rows || in->cols != out->cols) {
		return false;
	}

	u64 size = (u64) out->rows * out->cols;

	f32 sum = 0.0f;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = expf(in->data[i]);
		sum += in->data[i];
	}

	mat_scale(out, 1.0f/sum);

	return true;

}

b32 mat_cross_entropy(matrix* out, const matrix* correct, const matrix* prob) {
	if (correct->rows != prob->rows || correct->cols != prob->cols) {
		return false;
	}
	if (out->rows != prob->rows || out->cols != prob->cols) {
		return false;
	}

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = correct->data[i] > 0.0f ? -correct->data[i] * logf(fmaxf(prob->data[i], 1e-7f)) : 0.0f;
	}

	return true;

}

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
	if (out->rows != in->rows || out->cols != in->cols) {
		return false;
	}
	if (in->rows != grad->rows || in->cols != grad->cols) {
		return false;
	}

	u64 size = (u64) out->cols * out->rows;

	for (u64 i = 0; i < size; i++) {
		out->data[i] +=in->data[i] > 0.0f ? grad->data[i] : 0.0f;
	}

	return true;

}

	









