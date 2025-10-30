import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

balls = []
for i in range(N):
    balls.append(2 ** A[i])
    while len(balls) > 1 and balls[-1] == balls[-2]:
        balls.pop()
        balls.pop()
        balls.append(balls[-1] + balls[-1])
print(len(balls))