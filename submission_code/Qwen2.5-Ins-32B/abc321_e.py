# YOUR CODE HERE
import sys

def solve(N, X, K):
    if K == 0:
        return 1
    if X == 1:
        if K == 1:
            return 2
        elif K == 2:
            return N - 2
        else:
            return 0
    depth = 0
    x = X
    while x > 1:
        x = x // 2
        depth += 1
    if depth < K:
        return 0
    if depth == K:
        return 1
    remaining = K - (depth - 1)
    if remaining == 1:
        return 2
    elif remaining == 2:
        return 2 * (N // (2 ** (depth + 1))) - 2
    else:
        return 0

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(solve(N, X, K))