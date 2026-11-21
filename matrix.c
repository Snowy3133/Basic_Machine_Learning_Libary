
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

matrix* mat_load(memory_arena* arena, u32 rows, u32 cols, const char* filename) {
	FILE* file = fopen(filename, "rb");
	if (file == NULL) {
		fprintf(stderr, "mat_load: could not open %s\n", filename);
		return NULL;
	}

	// remember where the arena was, so a failed load can give the memory back
	u64 start_pos = arena->pos;

	matrix* mat = mat_create(arena, rows, cols);
	if (mat == NULL) {
		fprintf(stderr, "mat_load: arena too small for a %u x %u matrix\n", rows, cols);
		arena_pop_to(arena, start_pos);
		fclose(file);
		return NULL;
	}

	u64 count = (u64)rows * cols;
	u64 read = fread(mat->data, sizeof(f32), count, file);
	fclose(file);

	if (read != count) {
		fprintf(stderr, "mat_load: %s only has %llu values, expected %llu\n",
			filename, (unsigned long long)read, (unsigned long long)count);
		arena_pop_to(arena, start_pos);
		return NULL;
	}

	return mat;
}

void mat_copy(matrix* dst, const matrix* src) {
	ASSERT(dst->rows == src->rows && dst->cols == src->cols);

	memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);
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

void mat_scale_row(matrix* mat, u64 row_start, f32 scale) {
	u64 next_row = (u64) row_start + mat->cols;

	for (u64 i = row_start; i < next_row; i++) {
		mat->data[i] *= scale;
	}
}

f32 mat_sum(const matrix* mat) {
	u64 size = (u64) mat->cols * mat->rows;

	f32 sum = 0.0f;
	for (u64 i = 0; i < size; i++) {
		sum += mat->data[i];
	}

	return sum;

}

u64 mat_row_argmax(const matrix* mat, u64 row_start) {
	u64 next_row = row_start + mat->cols;
	
	u64 max_i = row_start;

	for (u64 i = row_start; i < next_row; i++) {
		if (mat->data[i] > mat->data[max_i]) {
			max_i = i;
		}
	}

	return max_i;

}

u64 mat_argmax(const matrix* mat) {
	u64 size = (u64) mat->cols * mat->rows;

	u64 max_i = 0;

	for (u64 i = 0; i < size; i++) {
		if (mat->data[i] > mat->data[max_i]) {
			max_i = i;
		}
	}

	return max_i;

}

void mat_add(matrix* out, const matrix* a, const matrix* b) {
	ASSERT(a->rows == b->rows && a->cols == b->cols);
	ASSERT(out->rows == a->rows && out->cols == a->cols);

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = a->data[i] + b->data[i];
	}
}

void mat_sub(matrix* out, const matrix* a, const matrix* b) {
	ASSERT(a->rows == b->rows && a->cols == b->cols);
	ASSERT(out->rows == a->rows && out->cols == a->cols);

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = a->data[i] - b->data[i];
	}
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

void mat_mul(matrix* out, matrix* a, matrix* b, b8 zero_out, b8 transpose_a, b8 transpose_b) {
#ifndef NDEBUG
    // shapes of a and b after any transpose; only needed for the checks
    u64 a_rows = transpose_a ? a->cols : a->rows;
    u64 a_cols = transpose_a ? a->rows : a->cols;
    u64 b_rows = transpose_b ? b->cols : b->rows;
    u64 b_cols = transpose_b ? b->rows : b->cols;

    ASSERT(a_cols == b_rows);
    ASSERT(out->rows == a_rows && out->cols == b_cols);
#endif
    ASSERT(out->data != a->data && out->data != b->data);

    if (zero_out) mat_fill(out, 0.0f);

    if      (!transpose_a && !transpose_b) _mat_mul_nn(out, a, b);
    else if (!transpose_a &&  transpose_b) _mat_mul_nt(out, a, b);
    else if ( transpose_a && !transpose_b) _mat_mul_tn(out, a, b);
    else                                   _mat_mul_tt(out, a, b);
}

void mat_relu(matrix* out, const matrix* in) {
	ASSERT(out->rows == in->rows && out->cols == in->cols);

	u64 size = (u64) out->rows * out->cols;
	for (u64 i = 0; i < size; i++) {
		out->data[i] = MAX(0, in->data[i]);
	}
}

void mat_softmax(matrix* out, const matrix* in) {
	ASSERT(out->rows == in->rows && out->cols == in->cols);

	u64 size = (u64) out->rows * out->cols;

	f32 sum = 0.0f;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = expf(in->data[i]);
		sum += in->data[i];
	}

	mat_scale(out, 1.0f/sum);
}

void mat_cross_entropy(matrix* out, const matrix* correct, const matrix* prob) {
	ASSERT(correct->rows == prob->rows && correct->cols == prob->cols);
	ASSERT(out->rows == prob->rows && out->cols == prob->cols);

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = correct->data[i] > 0.0f ? -correct->data[i] * logf(fmaxf(prob->data[i], 1e-7f)) : 0.0f;
	}
}

void mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
	ASSERT(out->rows == in->rows && out->cols == in->cols);
	ASSERT(grad->rows == in->rows && grad->cols == in->cols);

	u64 size = (u64) out->cols * out->rows;

	for (u64 i = 0; i < size; i++) {
		out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
	}
}


void mat_add_bias(matrix* out, const matrix* x, const matrix* bias) {
	ASSERT(out->rows == x->rows && out->cols == x->cols);
	ASSERT(bias->rows == 1 && bias->cols == out->cols);

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		out->data[i] = x->data[i] + bias->data[i % bias->cols];
	}
}

void mat_add_bias_grad(matrix* bias_grad, const matrix* grad) {
	ASSERT(bias_grad->rows == 1 && bias_grad->cols == grad->cols);

	u64 size = (u64) grad->rows * grad->cols;

	for (u64 i = 0; i < size; i++) {
		bias_grad->data[i % grad->cols] += grad->data[i];
	}
}

void mat_softmax_rows(matrix* out, const matrix* in) {
	ASSERT(out->rows == in->rows && out->cols == in->cols);

	u64 size = (u64) out->rows * out->cols;

	f32 sum = 0.0f;

	f32 row_max = 0.0f;

	for (u64 i = 0; i < size; i++) {
		if (i % out->cols == 0) {
			row_max = in->data[mat_row_argmax(in, i)];
		}

		out->data[i] = expf(in->data[i] - row_max);
		sum += out->data[i];

		if (i % out->cols == out->cols - 1) {
			mat_scale_row(out, i + 1 - out->cols, 1.0f / sum);
			sum = 0.0f;
		}
	}
}

void mat_softmax_ce(matrix* loss, matrix* prob, const matrix* logits, const matrix* labels) {
	ASSERT(labels->rows == logits->rows && labels->cols == logits->cols);
	ASSERT(prob->rows == logits->rows && prob->cols == logits->cols);
	ASSERT(loss->rows == 1 && loss->cols == 1);

	mat_softmax_rows(prob, logits);

	loss->data[0] = 0.0f;

	for (u64 row = 0; row < logits->rows; row++) {
		u64 start = row * logits->cols;

		f32 row_max = logits->data[mat_row_argmax(logits, start)];

		f32 sum = 0.0f;
		for (u64 col = 0; col < logits->cols; col++) {
			sum += expf(logits->data[start + col] - row_max);
		}
		f32 log_sum = logf(sum);

		for (u64 col = 0; col < logits->cols; col++) {
			u64 i = start + col;
			loss->data[0] -= labels->data[i] * (logits->data[i] - row_max - log_sum);
		}
	}

	mat_scale(loss, 1.0f / (f32)logits->rows);
}

void mat_softmax_ce_grad(matrix* logits_grad, const matrix* prob, const matrix* labels, const matrix* grad) {
	ASSERT(prob->rows == labels->rows && prob->cols == labels->cols);
	ASSERT(logits_grad->rows == labels->rows && logits_grad->cols == labels->cols);
	ASSERT(grad->rows == 1 && grad->cols == 1);

	u64 size = (u64) logits_grad->cols * logits_grad->rows;

	for (u64 i = 0; i < size; i++) {
		logits_grad->data[i] += (grad->data[0] / logits_grad->rows) * (prob->data[i] - labels->data[i]);
	}
}


























