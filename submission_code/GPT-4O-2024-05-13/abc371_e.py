# YOUR CODE HERE
def count_distinct_subsequences(N, A):
    total_sum = 0
    last_occurrence = {}
    current_sum = 0
    start = 0

    for end in range(N):
        if A[end] in last_occurrence and last_occurrence[A[end]] >= start:
            start = last_occurrence[A[end]] + 1
        last_occurrence[A[end]] = end
        current_sum += end - start + 1
        total_sum += current_sum

    return total_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(count_distinct_subsequences(N, A))