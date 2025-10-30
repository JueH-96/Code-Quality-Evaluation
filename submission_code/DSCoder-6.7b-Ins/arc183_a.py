import sys

def lexicographically_nth_sequence(N, K, n):
    sequence = []
    for i in range(1, N+1):
        sequence += [i] * K
    sequence = sequence[:n] + sequence[n+1:] + [sequence[n]]
    return sequence

def solve(N, K):
    S = (N-1) * N * K // 2
    if S % 2 == 0:
        mid = S // 2
    else:
        mid = S // 2 + 1
    for i in range(1, N*K+1):
        if (i-1) * i // 2 >= mid:
            return lexicographically_nth_sequence(N, K, i-1)

N, K = map(int, sys.stdin.readline().split())
print(*solve(N, K))