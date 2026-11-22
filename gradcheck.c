
#include "gradcheck.h"
#include "matrix.h"
#include "base.h"

typedef f64 (*gc_fn)(void* ctx);

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

f64 weighted_sum(const matrix* out, const matrix* g);
f32 rel_error(f32 a, f32 n);
