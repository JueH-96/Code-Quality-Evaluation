# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
P = list(map(int, data[1:]))

max_P = max(P[1:])
if P[0] > max_P:
    print(0)
else:
    print(max_P - P[0] + 1)