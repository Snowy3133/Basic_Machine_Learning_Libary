
#include <math.h>
#include "gradcheck.h"
#include "matrix.h"
#include "base.h"

f32 grad_check(matrix* x, const matrix* auto_diff_grad, gc_fn f, void* ctx, f32 h) {
	ASSERT(x->rows == auto_diff_grad->rows && x->cols == auto_diff_grad->cols);

	u64 size = (u64) x->rows * x->cols;
	
	f32 error = 0.0f;
	f32 max_error = 0.0f;
	
	f64 add_h_result = 0.0f;
	f64 sub_h_result = 0.0f;
	f64 grad = 0.0f;
	f32 original = 0.0f;

	for (u64 i = 0; i < size; i++) {
		original = x->data[i];

		x->data[i] += h;
		add_h_result = f(ctx);
		x->data[i] = original;

		x->data[i] -= h;
		sub_h_result = f(ctx);
		x->data[i] = original;
		
		grad = (add_h_result - sub_h_result) / (2 * h);

		error = rel_error(grad, auto_diff_grad->data[i]);
		if (isnan(error) || error > max_error) {
			max_error = error;
		}
	}

	return max_error;

}

f64 weighted_sum(const matrix* out, const matrix* g) {
	ASSERT(out->rows == g->rows && out->cols == g->cols);

	f64 sum = 0.0f;

	u64 size = (u64) out->rows * out->cols;

	for (u64 i = 0; i < size; i++) {
		sum += (f64) out->data[i] * g->data[i];
	}

	return sum;

}

f32 rel_error(f32 a, f32 n) {
	f32 difference = fabsf(a - n);
	f32 relative_difference = difference / MAX(MAX(fabsf(a), fabsf(n)), 1e-6f);

	return fabsf(relative_difference);

}
