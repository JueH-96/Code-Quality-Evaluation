# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = [tuple(map(int, data[i])) for i in range(3, 3 + 2 * Q, 2)]

# Precompute the number of consecutive pairs for each position
consecutive_pairs = [0] * N
for i in range(1, N):
    if S[i] == S[i - 1]:
        consecutive_pairs[i] = consecutive_pairs[i - 1] + 1
    else:
        consecutive_pairs[i] = consecutive_pairs[i - 1]

# Process each query
results = []
for l, r in queries:
    if l == 1:
        count = consecutive_pairs[r - 1]
    else:
        count = consecutive_pairs[r - 1] - consecutive_pairs[l - 1]
    results.append(count)

# Print the results
for result in results:
    print(result)