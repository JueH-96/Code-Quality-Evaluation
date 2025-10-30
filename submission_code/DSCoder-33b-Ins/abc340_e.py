# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = [0]*N

for i in range(N):
    boxes[i] = A[i]

for i in range(M):
    C = 0
    while boxes[B[i]] > 0:
        boxes[(B[i]+C)%N] += 1
        boxes[B[i]] -= 1
        C += 1

print(*boxes)