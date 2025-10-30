from collections import deque
from sys import stdin, stdout

def read_ints():
    return list(map(int, stdin.readline().split()))

def read_int():
    return int(stdin.readline())

def read_str():
    return stdin.readline().strip()

def solve():
    H, W = read_ints()
    A = [read_ints() for _ in range(H)]
    B = [read_ints() for _ in range(H)]

    def is_valid(a, b):
        for i in range(H):
            for j in range(W):
                if a[i][j] != b[i][j]:
                    return False
        return True

    def swap_rows(a, i):
        a[i], a[i + 1] = a[i + 1], a[i]

    def swap_cols(a, i):
        for row in a:
            row[i], row[i + 1] = row[i + 1], row[i]

    queue = deque([(A, 0)])
    visited = {tuple(tuple(row) for row in A)}
    while queue:
        a, steps = queue.popleft()
        if is_valid(a, B):
            return steps
        for i in range(H - 1):
            b = [row[:] for row in a]
            swap_rows(b, i)
            b_tuple = tuple(tuple(row) for row in b)
            if b_tuple not in visited:
                queue.append((b, steps + 1))
                visited.add(b_tuple)
        for i in range(W - 1):
            b = [row[:] for row in a]
            swap_cols(b, i)
            b_tuple = tuple(tuple(row) for row in b)
            if b_tuple not in visited:
                queue.append((b, steps + 1))
                visited.add(b_tuple)
    return -1

print(solve())