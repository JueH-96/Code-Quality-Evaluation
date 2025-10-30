import sys

def solve(N, M, A):
    A.sort()
    total = 0
    for i, a in enumerate(A):
        total += a
        if total > M:
            return a - 1 if i > 0 else 'infinite' if total - a <= M else a
    return A[-1] if total <= M else A[-1] - 1

input = sys.stdin.read
data = input().split()
N, M = map(int, (data[0], data[1]))
A = list(map(int, data[2:]))
print(solve(N, M, A))