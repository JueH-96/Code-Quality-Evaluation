# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Initialize boxes with initial ball counts
boxes = A[:]

# Process each operation
for i in range(M):
    box_idx = B[i]
    balls_to_distribute = boxes[box_idx]
    boxes[box_idx] = 0  # Remove all balls from this box
    
    if balls_to_distribute == 0:
        continue
    
    # Calculate how many complete cycles and remaining balls
    complete_cycles = balls_to_distribute // N
    remaining_balls = balls_to_distribute % N
    
    # Add complete cycles to all boxes
    for j in range(N):
        boxes[j] += complete_cycles
    
    # Distribute remaining balls one by one
    for j in range(remaining_balls):
        target_box = (box_idx + 1 + j) % N
        boxes[target_box] += 1

# Output the result
print(*boxes)