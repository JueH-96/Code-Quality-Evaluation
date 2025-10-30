# YOUR CODE HERE

from bisect import bisect_left, bisect_right

def solve():
    N = int(input())
    X = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = int(input())

    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + P[i]

    for _ in range(Q):
        L, R = map(int, input().split())
        left = bisect_left(X, L)
        right = bisect_right(X, R)
        print(prefix_sum[right] - prefix_sum[left])

solve()