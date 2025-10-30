# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Process each operation
for i in range(M):
    box_idx = B[i]
    balls = A[box_idx]
    A[box_idx] = 0
    
    # Distribute balls one by one
    for j in range(balls):
        target_box = (box_idx + j + 1) % N
        A[target_box] += 1

print(' '.join(map(str, A)))