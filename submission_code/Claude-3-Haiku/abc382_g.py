from typing import Tuple

def min_moves(K: int, S_x: int, S_y: int, T_x: int, T_y: int) -> int:
    def get_tile(x: int, y: int) -> Tuple[int, int, int]:
        i = x // K
        j = y // K
        k = (y - j * K) % K
        return i, j, k

    def is_adjacent(i1: int, j1: int, k1: int, i2: int, j2: int, k2: int) -> bool:
        if (i1, j1) == (i2, j2):
            return abs(k1 - k2) == 1
        if (i1 - i2) ** 2 + (j1 - j2) ** 2 == 1:
            return True
        return False

    start_tile = get_tile(S_x, S_y)
    end_tile = get_tile(T_x, T_y)

    queue = [(start_tile, 0)]
    visited = set([start_tile])

    while queue:
        curr_tile, moves = queue.pop(0)
        if curr_tile == end_tile:
            return moves

        for i, j, k in [(curr_tile[0] + 1, curr_tile[1], curr_tile[2]),
                       (curr_tile[0] - 1, curr_tile[1], curr_tile[2]),
                       (curr_tile[0], curr_tile[1] + 1, curr_tile[2]),
                       (curr_tile[0], curr_tile[1] - 1, curr_tile[2])]:
            if is_adjacent(curr_tile[0], curr_tile[1], curr_tile[2], i, j, k) and (i, j, k) not in visited:
                queue.append(((i, j, k), moves + 1))
                visited.add((i, j, k))

    return -1

T = int(input())
for _ in range(T):
    K, S_x, S_y, T_x, T_y = map(int, input().split())
    print(min_moves(K, S_x, S_y, T_x, T_y))