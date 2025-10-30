import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Create a dictionary to store the last occurrence of each value
last_occurrence = {}
# Create a list to store the number of distinct values up to each index
distinct_count = [0] * N

# Iterate through the array and populate the last_occurrence and distinct_count lists
for i in range(N):
    if A[i] in last_occurrence:
        distinct_count[i] = distinct_count[last_occurrence[A[i]]]
    else:
        distinct_count[i] = i
    last_occurrence[A[i]] = i

# Calculate the sum of f(i, j) for all i and j
total_sum = 0
for i in range(N):
    total_sum += (i - distinct_count[i] + 1) * (N - i)

print(total_sum)