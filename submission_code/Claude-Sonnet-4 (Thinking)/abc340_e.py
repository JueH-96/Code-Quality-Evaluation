N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = A[:]

for i in range(M):
    box_idx = B[i]
    count = boxes[box_idx]
    boxes[box_idx] = 0
    
    if count == 0:
        continue
    
    # Each box gets at least count // N balls
    base_balls = count // N
    extra_balls = count % N
    
    for j in range(N):
        boxes[j] += base_balls
    
    # The first extra_balls positions get one extra ball
    for c in range(1, extra_balls + 1):
        target_box = (box_idx + c) % N
        boxes[target_box] += 1

print(' '.join(map(str, boxes)))