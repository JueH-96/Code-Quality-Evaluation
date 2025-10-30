# YOUR CODE HERE
n = int(input())
positions = list(map(int, input().split()))

A = []
for i in range(1, n + 1):
    pos = positions[i - 1] - 1  # Convert to 0-indexed
    A.insert(pos, i)

print(' '.join(map(str, A)))