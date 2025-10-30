# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    count = defaultdict(int)
    distinct_count = 0
    total_sum = 0
    j = 0

    for i in range(N):
        while j < N and count[A[j]] == 0:
            count[A[j]] += 1
            distinct_count += 1
            j += 1

        total_sum += distinct_count
        count[A[i]] -= 1
        if count[A[i]] == 0:
            distinct_count -= 1

    print(total_sum)

solve()