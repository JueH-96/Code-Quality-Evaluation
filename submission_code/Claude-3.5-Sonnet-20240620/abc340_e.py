# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = A.copy()

for i in range(M):
    C = 0
    balls = boxes[B[i]]
    boxes[B[i]] = 0
    
    while balls > 0:
        C += 1
        target_box = (B[i] + C) % N
        boxes[target_box] += 1
        balls -= 1

print(*boxes)