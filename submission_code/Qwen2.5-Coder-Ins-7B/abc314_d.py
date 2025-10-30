# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
Q = int(data[2])

for i in range(Q):
    t = int(data[3 + 3 * i])
    if t == 1:
        x = int(data[4 + 3 * i]) - 1
        c = data[5 + 3 * i]
        S = S[:x] + c + S[x+1:]
    elif t == 2:
        S = S.lower()
    elif t == 3:
        S = S.upper()

print(S)