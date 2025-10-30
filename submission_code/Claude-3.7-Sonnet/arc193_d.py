from collections import deque

def min_operations(n, A, B):
    # Check if there are enough pieces
    pieces_A = A.count('1')
    pieces_B = B.count('1')
    if pieces_A < pieces_B:
        return -1  # Impossible if A has fewer pieces than B needs
    
    # Convert A to a state array (where state[i] is the number of pieces in square i+1)
    initial_state = [0] * n
    for i in range(n):
        if A[i] == '1':
            initial_state[i] = 1
    
    # BFS
    queue = deque([(initial_state, 0)])  # (state, operations)
    visited = {tuple(initial_state)}
    
    while queue:
        state, ops = queue.popleft()
        
        # Check if the state satisfies the condition
        valid = True
        for i in range(n):
            if (state[i] > 0 and B[i] == '0') or (state[i] == 0 and B[i] == '1'):
                valid = False
                break
                
        if valid:
            return ops
        
        # Try all possible choices for the operation
        for i in range(n):
            # Compute the new state after the operation
            new_state = [0] * n
            for j in range(n):
                if j < i:
                    new_state[j+1] += state[j]  # Move right
                elif j > i:
                    new_state[j-1] += state[j]  # Move left
                else:
                    new_state[j] += state[j]  # Stay in the same position
            
            # Check if we've visited this state
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in visited:
                queue.append((new_state, ops + 1))
                visited.add(new_state_tuple)
    
    return -1  # No solution found

# Main function to read input and process each test case
def main():
    T = int(input())
    for _ in range(T):
        n = int(input())
        A = input()
        B = input()
        print(min_operations(n, A, B))

if __name__ == "__main__":
    main()