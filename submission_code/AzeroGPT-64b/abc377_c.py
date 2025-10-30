from collections import defaultdict

def read_ints():
    return map(int, input().split())

def solve():
    N, M = read_ints()
    A = [None] * M
    safe_count = N * N - M
    safe_cells, captures = defaultdict(int), defaultdict(int)
    for i in range(M):
        a, b = read_ints()
        A[i] = (a, b)
        key = (a + b) % 4
        safe_cells[key] += 1

    for row, col in A:
        for dr, dc in ((-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)):
            R, C = row + dr, col + dc
            if 0 < R <= N and 0 < C <= N:
                r, c = R, C
                if (r, c) not in A:
                    key = (R + C) % 4
                    captures[key] += 1

    for k in range(4):
        safe_count -= max(0, captures[k] - safe_cells[k])

    return safe_count

print(solve())