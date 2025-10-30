def min_operations_to_empty_boxes(N, X, A, B, P, Q):
    # Adjust X to be zero-indexed
    X -= 1
    
    # Initialize the number of operations
    operations = 0
    
    # Create a list to track which boxes can be emptied
    can_empty = [False] * N
    
    # Mark the target box as emptyable
    can_empty[X] = True
    
    # Create a queue for processing boxes
    from collections import deque
    queue = deque([X])
    
    # Process the boxes
    while queue:
        current_box = queue.popleft()
        
        # Check which boxes can be emptied by taking balls from the current box
        for i in range(N):
            if P[i] - 1 == current_box or Q[i] - 1 == current_box:
                if not can_empty[i]:
                    can_empty[i] = True
                    queue.append(i)
    
    # Count the operations needed to empty all boxes except X
    for i in range(N):
        if i != X and (A[i] > 0 or B[i] > 0):
            if not can_empty[i]:
                return -1  # If we cannot empty this box, return -1
            operations += 1  # We can empty this box, count the operation
    
    return operations

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))
P = list(map(int, data[2*N+2:3*N+2]))
Q = list(map(int, data[3*N+2:4*N+2]))

# Get the result
result = min_operations_to_empty_boxes(N, X, A, B, P, Q)

# Print the result
print(result)