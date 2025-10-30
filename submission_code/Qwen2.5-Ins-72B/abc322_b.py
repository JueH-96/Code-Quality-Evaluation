# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

if S == T[:N] and S == T[-N:]:
    print(0)
elif S == T[:N]:
    print(1)
elif S == T[-N:]:
    print(2)
else:
    print(3)