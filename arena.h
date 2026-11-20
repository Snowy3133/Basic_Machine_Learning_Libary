

#ifndef ARENA_H
#define ARENA_H

#include "base.h"

#define PUSH_STRUCT(arena, T) 			(T*)arena_push(arena, sizeof(T), false)
#define PUSH_STRUCT_NOT_ZEROED(arena, T) 	(T*)arena_push(arena, sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) 		(T*)arena_push(arena, sizeof(T) * (n), false)
#define PUSH_ARRAY_NOT_ZEROED(arena, T, n) 	(T*)arena_push(arena, sizeof(T) * (n), true)

typedef struct {
	u64 capacity;
	u64 pos;
} memory_arena;

memory_arena* arena_create(u64 capacity);
void arena_destroy(memory_arena* arena);
void* arena_push(memory_arena* arena, u64 size, b32 non_zero);
void arena_pop(memory_arena* arena, u64 size);
void arena_pop_to(memory_arena* arena, u64 pos);
void arena_clear(memory_arena*); 


#endif

