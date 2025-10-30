from collections import deque
import sys
input = sys.stdin.readline

Q = int(input())

A = deque()

for _ in range(Q):
    q = list(map(int, input().split()))
    if q[0] == 1:
        A.append(q[1])
    else:
        k = q[1]
        print(A[-k])