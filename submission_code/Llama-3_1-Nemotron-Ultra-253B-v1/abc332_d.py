import itertools

def count_inversion(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv

H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]

min_ops = None

for row_perm in itertools.permutations(range(H)):
    reordered_rows = [A[i] for i in row_perm]
    for col_perm in itertools.permutations(range(W)):
        transformed = [[row[j] for j in col_perm] for row in reordered_rows]
        if transformed == B:
            current = count_inversion(row_perm) + count_inversion(col_perm)
            if min_ops is None or current < min_ops:
                min_ops = current

print(min_ops if min_ops is not None else -1)