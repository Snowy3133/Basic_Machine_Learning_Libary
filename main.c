

#include "arena.c"
#include "matrix.c"

int main(void) {
    memory_arena* perm = arena_create(MiB(64));
    if (perm == NULL) {
        fprintf(stderr, "failed to allocate arena\n");
        return 1;
    }

    // quick tests go here

    arena_destroy(perm);
    return 0;
}
