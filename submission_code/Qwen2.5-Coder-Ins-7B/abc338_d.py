# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X = list(map(int, data[2:]))

# Calculate the total length of the tour without any bridge closed
total_length = 0
for i in range(M - 1):
    total_length += abs(X[i] - X[i + 1])

# Calculate the minimum possible length of the tour when one bridge is closed
min_length = float('inf')
for i in range(N):
    if i == 0:
        length = abs(X[-1] - X[0]) + total_length - abs(X[-1] - X[1])
    elif i == N - 1:
        length = abs(X[-2] - X[-1]) + total_length - abs(X[-2] - X[0])
    else:
        length = total_length - abs(X[i - 1] - X[i]) + abs(X[i - 1] - X[i + 1])
    min_length = min(min_length, length)

print(min_length)