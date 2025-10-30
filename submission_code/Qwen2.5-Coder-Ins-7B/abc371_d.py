# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:N+1]))
P = list(map(int, data[N+1:2*N+1]))
Q = int(data[2*N+1])
queries = [(int(data[2*N+2+2*i]), int(data[2*N+3+2*i])) for i in range(Q)]

# Create a list of tuples (X_i, P_i)
villages = list(zip(X, P))

# Sort the villages by their coordinates
villages.sort()

# Create a list of prefix sums of the number of villagers
prefix_sum = [0]
for _, p in villages:
    prefix_sum.append(prefix_sum[-1] + p)

# Answer each query
for L, R in queries:
    # Find the indices of the villages with coordinates just before and after L and R
    i = bisect.bisect_left(villages, (L, float('-inf')))
    j = bisect.bisect_right(villages, (R, float('inf')))
    # The number of villagers in the range [L, R] is the difference in prefix sums
    print(prefix_sum[j] - prefix_sum[i])