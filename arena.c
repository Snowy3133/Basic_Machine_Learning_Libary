
#include <stdlib.h>
#include <string.h>
#include "arena.h"

#define ARENA_BASE_POS (sizeof(memory_arena))
#define ARENA_ALIGN (sizeof(void*))

#define ALIGN_UP_POW_TWO(n,p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

memory_arena* arena_create(u64 capacity){
	memory_arena* arena = malloc(capacity);

	arena -> capacity = capacity;
	arena -> pos = ARENA_BASE_POS;

	return arena;
}

void arena_destroy(memory_arena* arena){
	free(arena);
}

void* arena_push(memory_arena* arena, u64 size, b32 non_zero){
	u64 pos_aligned = ALIGN_UP_POW_TWO(arena->pos, ARENA_ALIGN);
	u64 new_pos = pos_aligned + size;
	
	if (new_pos > arena->capacity){
		return NULL;
	}

	arena->pos = new_pos;

	u8* out = (u8*)arena + pos_aligned;

	if (!non_zero){
		memset(out, 0 , size);
	}

	return out;
}

void arena_pop(memory_arena* arena, u64 size) {
	size = MIN(size, arena->pos - ARENA_BASE_POS);
	arena->pos -= size;

}

void arena_pop_to(memory_arena* arena, u64 pos) {
	u64 size = pos < arena->pos ? arena->pos - pos : 0;
	arena_pop(arena, size);

}

void arena_clear(memory_arena* arena) {
	arena_pop_to(arena, ARENA_BASE_POS);
}



