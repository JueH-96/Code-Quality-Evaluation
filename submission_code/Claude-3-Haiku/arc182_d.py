from collections import deque

def min_operations(N, M, A, B):
    # Initialize a queue to store the current sequence and the number of operations performed
    queue = deque([(A, 0)])
    
    # Set to keep track of visited sequences
    visited = set()
    
    while queue:
        curr_A, ops = queue.popleft()
        
        # Check if the current sequence is equal to the target sequence B
        if curr_A == B:
            return ops
        
        # Add the current sequence to the visited set
        visited.add(tuple(curr_A))
        
        # Try all possible operations on the current sequence
        for i in range(N):
            for op in [1, -1]:
                new_A = curr_A[:]
                new_A[i] = (new_A[i] + op) % M
                
                # Check if the new sequence is a good sequence
                if all(new_A[j] != new_A[j+1] for j in range(N-1)):
                    # Add the new sequence to the queue if it hasn't been visited before
                    if tuple(new_A) not in visited:
                        queue.append((new_A, ops + 1))
    
    # If the goal is unachievable, return -1
    return -1

# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve the problem and print the result
print(min_operations(N, M, A, B))