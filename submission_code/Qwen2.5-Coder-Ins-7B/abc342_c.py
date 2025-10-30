# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
Q = int(data[2])

operations = []
for i in range(Q):
    c = data[3 + 2 * i]
    d = data[4 + 2 * i]
    operations.append((c, d))

for c, d in operations:
    S = S.replace(c, d)

print(S)