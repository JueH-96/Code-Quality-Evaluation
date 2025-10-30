import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % M

    from collections import Counter
    count = Counter(prefix_sum)
    result = 0
    for c in count.values():
        result += c * (c - 1) // 2

    print(result)

solve()