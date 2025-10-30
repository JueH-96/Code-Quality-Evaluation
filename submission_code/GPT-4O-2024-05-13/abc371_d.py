# YOUR CODE HERE
import sys
import bisect

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:N+1]))
P = list(map(int, data[N+1:2*N+1]))
Q = int(data[2*N+1])
queries = data[2*N+2:]

# Create a prefix sum array for the villagers
prefix_sum = [0] * (N + 1)
for i in range(1, N + 1):
    prefix_sum[i] = prefix_sum[i - 1] + P[i - 1]

# Process each query
output = []
for i in range(Q):
    L = int(queries[2 * i])
    R = int(queries[2 * i + 1])
    
    # Find the range of indices in X that fall within [L, R]
    left_index = bisect.bisect_left(X, L)
    right_index = bisect.bisect_right(X, R) - 1
    
    if left_index <= right_index:
        total_villagers = prefix_sum[right_index + 1] - prefix_sum[left_index]
    else:
        total_villagers = 0
    
    output.append(total_villagers)

# Print the results
for result in output:
    print(result)