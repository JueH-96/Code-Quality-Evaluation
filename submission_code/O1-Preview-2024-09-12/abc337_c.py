# YOUR CODE HERE
import sys

lines = sys.stdin.read().split()
N = int(lines[0])
A = list(map(int, lines[1:N+1]))

successor = {}
front = None

for i in range(1,N+1):
    A_i = A[i-1]
    if A_i == -1:
        front = i
    else:
        successor[A_i] = i

line = []

current = front

while True:
    line.append(current)
    if current in successor:
        current = successor[current]
    else:
        break

print(' '.join(map(str, line)))