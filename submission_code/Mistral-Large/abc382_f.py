import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])

bars = []
for i in range(N):
    R = int(data[3 + 3 * i])
    C = int(data[4 + 3 * i])
    L = int(data[5 + 3 * i])
    bars.append((R, C, L))

# Sort bars by their initial row and then by column
bars.sort()

# Initialize a queue for each column to keep track of the highest row a bar can reach
columns = [deque() for _ in range(W + 1)]

for R, C, L in bars:
    max_row = H
    for j in range(C, C + L):
        while columns[j] and columns[j][0] < R:
            columns[j].popleft()
        if columns[j]:
            max_row = min(max_row, columns[j][0] - 1)
    final_row = max_row
    for j in range(C, C + L):
        while columns[j] and columns[j][-1] >= final_row:
            columns[j].pop()
        columns[j].append(final_row)
    print(final_row)