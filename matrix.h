
#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"

typedef struct {
	u32 rows, cols;

	f32* data;
} matrix;

matrix* mat_create(memory_arena* arena, u32 rows, u32 cols);
matrix* mat_load(memory_arena* arena, u32 rows, u32 cols, const char* filename);
void mat_copy(matrix* dst, const matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 fill);
void mat_fill_random(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
void mat_scale_row(matrix* mat, u64 row_start, f32 scale);
f32 mat_sum(const matrix* mat);
u64 mat_row_argmax(const matrix* mat, u64 row_start);
u64 mat_argmax(const matrix* mat);
void mat_add(matrix* out, const matrix* a, const matrix* b);
void mat_sub(matrix* out, const matrix* a, const matrix* b);
void mat_mul(matrix* out, matrix* a, matrix* b, b8 zero_out, b8 transpose_a, b8 transpose_b);
void mat_relu(matrix* out, const matrix* in);
void mat_softmax(matrix* out, const matrix* in);
void mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
void mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
void mat_add_bias(matrix* out, const matrix* x, const matrix* bias);
void mat_add_bias_grad(matrix* bias_grad, const matrix* grad);
void mat_softmax_rows(matrix* out, const matrix* in);
void mat_softmax_ce(matrix* loss, matrix* prob, const matrix* logits, const matrix* labels);
void mat_softmax_ce_grad(matrix* logits_grad, const matrix* prob, const matrix* labels, const matrix* grad);

#endif

