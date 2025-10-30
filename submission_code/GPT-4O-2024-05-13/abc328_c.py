# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]

queries = []
for i in range(Q):
    l = int(data[3 + 2 * i])
    r = int(data[4 + 2 * i])
    queries.append((l, r))

# Precompute the number of double letters up to each position
double_count = [0] * (N + 1)
for i in range(1, N):
    double_count[i] = double_count[i - 1]
    if S[i - 1] == S[i]:
        double_count[i] += 1

# Answer each query
results = []
for l, r in queries:
    results.append(double_count[r - 1] - double_count[l - 1])

# Print results
for result in results:
    print(result)