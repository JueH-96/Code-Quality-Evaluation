# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
giants = []

for i in range(1, 2 * N, 2):
    A = int(data[i])
    B = int(data[i + 1])
    giants.append((A, B))

giants.sort(key=lambda x: x[1] - x[0], reverse=True)

max_height = 0
current_height = 0

for A, B in giants:
    current_height += A
    max_height = max(max_height, current_height + B)

print(max_height)