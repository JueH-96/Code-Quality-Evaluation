import sys
import bisect

input = sys.stdin.read
data = input().split()

index = 0

# Read number of villages
N = int(data[index])
index += 1

# Read village coordinates
X = list(map(int, data[index:index+N]))
index += N

# Read population of each village
P = list(map(int, data[index:index+N]))
index += N

# Read number of queries
Q = int(data[index])
index += 1

# Prepare to answer the queries
results = []

# We will use a prefix sum array to efficiently calculate the sum of populations in a range
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + P[i]

# Process each query
for _ in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    index += 2
    
    # Find the range of indices in X that are between L and R
    left_index = bisect.bisect_left(X, L)
    right_index = bisect.bisect_right(X, R) - 1
    
    if left_index <= right_index:
        # Calculate the sum of populations in the range
        total_population = prefix_sum[right_index + 1] - prefix_sum[left_index]
    else:
        total_population = 0
    
    results.append(total_population)

# Print all results
for result in results:
    print(result)