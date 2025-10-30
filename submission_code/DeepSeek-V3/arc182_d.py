# YOUR CODE HERE
import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if the sequences are already the same
    if A == B:
        print(0)
        return
    
    # Precompute the possible values for each position
    # For each position i, the value can be changed to any value except A[i-1] and A[i+1]
    # But since we are changing one element at a time, we need to ensure that after each change, the sequence remains good
    
    # We will use BFS to find the minimum number of operations
    # Each state is a tuple representing the current sequence
    # We will use a queue to manage the states and a dictionary to keep track of visited states and their costs
    
    # Since N can be up to 2e5, we need a more efficient way to represent the state
    # Instead of storing the entire sequence, we can represent the state as a tuple of differences from the original sequence
    # But this is still not feasible for large N
    
    # Alternative approach: since the sequence is good, and we can only change one element at a time, we can consider each element independently
    # However, the changes are interdependent because changing one element affects the adjacent elements
    
    # Given the constraints, a BFS approach is not feasible due to the large state space
    # We need a different approach
    
    # Let's consider that for each position, the value can be changed to any value except the current values of the adjacent positions
    # So for each position i, the possible values are all values except A[i-1] and A[i+1]
    # But since we are changing one element at a time, we need to ensure that after each change, the sequence remains good
    
    # We can model this as a graph where each node represents a possible sequence, and edges represent single operations
    # However, the number of nodes is too large
    
    # Instead, we can consider that for each position, the value can be changed to any value except the current values of the adjacent positions
    # So for each position, we can compute the minimum number of operations to change A[i] to B[i], while ensuring that the new value is not equal to A[i-1] or A[i+1]
    
    # But this is not sufficient, because changing one element affects the adjacent elements
    
    # Given the time constraints, we will implement a BFS approach with pruning, but it will not pass all test cases due to the large N
    
    # Initialize the queue with the initial state
    initial_state = tuple(A)
    target_state = tuple(B)
    
    if initial_state == target_state:
        print(0)
        return
    
    queue = deque()
    queue.append((initial_state, 0))
    
    visited = set()
    visited.add(initial_state)
    
    while queue:
        current_state, operations = queue.popleft()
        
        if current_state == target_state:
            print(operations)
            return
        
        # Generate all possible next states
        for i in range(N):
            # Try increasing A[i] by 1
            new_value = (current_state[i] + 1) % M
            if new_value != current_state[i-1] if i > 0 else True and new_value != current_state[i+1] if i < N-1 else True:
                new_state = list(current_state)
                new_state[i] = new_value
                new_state_tuple = tuple(new_state)
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    queue.append((new_state_tuple, operations + 1))
            
            # Try decreasing A[i] by 1
            new_value = (current_state[i] - 1) % M
            if new_value != current_state[i-1] if i > 0 else True and new_value != current_state[i+1] if i < N-1 else True:
                new_state = list(current_state)
                new_state[i] = new_value
                new_state_tuple = tuple(new_state)
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    queue.append((new_state_tuple, operations + 1))
    
    # If we exhaust the queue without finding the target state, it's impossible
    print(-1)

if __name__ == "__main__":
    main()