# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = int(data[1])
Q = int(data[2])
D = list(map(int, data[3:]))

min_cost = P
for i in range(N):
    min_cost = min(min_cost, Q + D[i])

print(min_cost)