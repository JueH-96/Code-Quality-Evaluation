# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

# Sort boxes by price
boxes = sorted(zip(A, B), key=lambda x: x[0])

# Try to find M boxes that satisfy the condition
total_cost = 0
for i in range(M):
    if boxes[i][1] >= B[i]:
        total_cost += boxes[i][0]
    else:
        total_cost = -1
        break

print(total_cost)