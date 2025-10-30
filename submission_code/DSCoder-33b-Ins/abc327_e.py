# YOUR CODE HERE
import sys

def max_rating():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    P.sort(reverse=True)
    sum_P = sum(P)
    sum_weighted_P = sum(P[i] * (0.9 ** i) for i in range(N))
    max_rating = -float('inf')

    for k in range(1, N+1):
        rating = (sum_weighted_P / sum(0.9 ** i for i in range(k))) - (1200 / k ** 0.5)
        max_rating = max(max_rating, rating)

    print(max_rating)

max_rating()