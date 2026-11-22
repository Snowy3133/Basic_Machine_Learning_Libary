


#ifndef GRADCHECK_H
#define GRADCHECK_H

#include "matrix.h"

typedef f64 (*gc_fn)(void* ctx);

f32 grad_check(matrix* x, const matrix* auto_diff_grad, gc_fn f, void* ctx, f32 h);
f64 weighted_sum(const matrix* out, const matrix* g);
f32 rel_error(f32 a, f32 n);

#endif
