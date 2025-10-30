# YOUR CODE HERE
import sys
import bisect

input = sys.stdin.read().split()
N = int(input[0])
X = list(map(int, input[1:N+1]))
P = list(map(int, input[N+1:2*N+1]))
Q = int(input[2*N+1])
queries = [(int(input[2*N+2+2*i]), int(input[2*N+3+2*i])) for i in range(Q)]

# Create a prefix sum array for the number of villagers
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + P[i]

# Process each query
results = []
for L, R in queries:
    left_index = bisect.bisect_left(X, L)
    right_index = bisect.bisect_right(X, R) - 1
    if left_index > right_index:
        results.append(0)
    else:
        results.append(prefix_sum[right_index + 1] - prefix_sum[left_index])

# Print all results
for result in results:
    print(result)