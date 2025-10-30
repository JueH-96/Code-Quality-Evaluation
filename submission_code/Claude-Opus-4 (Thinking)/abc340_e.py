N, M = map(int, input().split())
initial = list(map(int, input().split()))
B = list(map(int, input().split()))

global_add = 0
extra = [0] * N

for i in range(M):
    box_idx = B[i]
    
    # Calculate current balls in the box
    balls = initial[box_idx] + global_add + extra[box_idx]
    
    # Clear the box
    initial[box_idx] = 0
    extra[box_idx] = -global_add
    
    # Distribute balls
    full_rounds = balls // N
    remaining = balls % N
    
    global_add += full_rounds
    
    for j in range(1, remaining + 1):
        target_box = (box_idx + j) % N
        extra[target_box] += 1

# Calculate final values
result = []
for i in range(N):
    result.append(initial[i] + global_add + extra[i])

print(' '.join(map(str, result)))