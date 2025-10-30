# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
S = list(data[1])
Q = int(data[2])

operations = []
for i in range(Q):
    t, x, c = int(data[3 + 3 * i]), int(data[4 + 3 * i]), data[5 + 3 * i]
    operations.append((t, x, c))

for t, x, c in operations:
    if t == 1:
        S[x - 1] = c
    elif t == 2:
        S = [char.lower() for char in S]
    elif t == 3:
        S = [char.upper() for char in S]

print(''.join(S))