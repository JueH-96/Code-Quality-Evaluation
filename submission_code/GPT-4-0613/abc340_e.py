# YOUR CODE HERE
import sys
from collections import deque

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

boxes = [deque() for _ in range(N)]
for i in range(N):
    boxes[i].append(A[i])

for i in range(M):
    box_num = B[i]
    balls_in_hand = boxes[box_num].popleft()
    C = 0
    while balls_in_hand > 0:
        boxes[(box_num + C) % N].appendleft(balls_in_hand)
        balls_in_hand -= 1
        C += 1

for i in range(N):
    print(sum(boxes[i]), end=' ')
print()