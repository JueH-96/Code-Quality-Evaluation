N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = A.copy()

for i in range(M):
    b = B[i]
    balls = boxes[b]
    boxes[b] = 0
    
    c = 0
    while balls > 0:
        c += 1
        target = (b + c) % N
        boxes[target] += 1
        balls -= 1

print(*boxes)