from itertools import permutations

def evaluate(funcs, seq, x):
    for i in seq:
        a, b = funcs[i]
        x = a * x + b
    return x

n, k = map(int, input().split())
funcs = []
for _ in range(n):
    a, b = map(int, input().split())
    funcs.append((a, b))

ans = 0
for p in permutations(range(n), k):
    val = evaluate(funcs, p, 1)
    ans = max(ans, val)

print(ans)