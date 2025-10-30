# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Create a list of tuples (value, index)
indexed_P = [(P[i], i) for i in range(N)]
indexed_P.sort()

# Initialize variables
min_diff = float('inf')
i1 = 0

# Iterate through the sorted list to find the minimum difference
for i2 in range(K - 1, N):
    if indexed_P[i2][0] - indexed_P[i2 - K + 1][0] == K - 1:
        min_diff = min(min_diff, indexed_P[i2][1] - indexed_P[i2 - K + 1][1])

print(min_diff)