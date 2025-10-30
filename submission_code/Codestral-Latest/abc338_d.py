import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])
x = list(map(int, data[2:]))

# Calculate the minimum length of the tour when no bridge is closed
min_length = 0
for i in range(m - 1):
    min_length += min(abs(x[i] - x[i + 1]), n - abs(x[i] - x[i + 1]))

# Calculate the minimum length of the tour when each bridge is closed
min_length_closed = float('inf')
for i in range(1, n):
    length_closed = 0
    for j in range(m - 1):
        if (x[j] <= i and x[j + 1] <= i) or (x[j] > i and x[j + 1] > i):
            length_closed += min(abs(x[j] - x[j + 1]), n - abs(x[j] - x[j + 1]))
        else:
            length_closed += min(abs(x[j] - i) + abs(i - x[j + 1]), n - abs(x[j] - i) + n - abs(i - x[j + 1]))
    min_length_closed = min(min_length_closed, length_closed)

# The answer is the minimum of the two lengths
print(min(min_length, min_length_closed))