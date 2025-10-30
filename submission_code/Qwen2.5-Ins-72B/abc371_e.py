# YOUR CODE HERE
from collections import defaultdict

def count_distinct_subarrays(N, A):
    result = 0
    for i in range(N):
        seen = defaultdict(int)
        distinct_count = 0
        for j in range(i, N):
            if seen[A[j]] == 0:
                distinct_count += 1
            seen[A[j]] += 1
            result += distinct_count
    return result

N = int(input())
A = list(map(int, input().split()))

print(count_distinct_subarrays(N, A))