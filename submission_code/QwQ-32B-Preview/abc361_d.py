from collections import deque

def main():
    import sys
    N_str, S, T = sys.stdin.read().split()
    N = int(N_str)
    cells = list(S) + ['.', '.']
    target = list(T) + ['.', '.']
    
    # Check if frequency of colors matches
    if sorted(S) != sorted(T):
        print(-1)
        return
    
    # Find positions of empty cells
    empty = (N, N+1)
    
    # Initialize BFS
    queue = deque()
    visited = set()
    initial_state = tuple(cells)
    queue.append((initial_state, empty, 0))
    visited.add(initial_state)
    
    while queue:
        current_cells, current_empty, operations = queue.popleft()
        
        # Check if current state matches the target
        if current_cells == tuple(target):
            print(operations)
            return
        
        # Identify empty cells
        k, k1 = current_empty
        if k1 != k + 1:
            print(-1)
            return
        
        # Try to move stones from x and x+1 to k and k+1 for all possible x
        for x in range(1, N+1):
            x0, x1 = x-1, x  # Adjust for zero-based indexing
            if current_cells[x0] != '.' and current_cells[x1] != '.':
                # Create a new state by moving stones from x and x+1 to k and k+1
                new_cells = list(current_cells)
                new_cells[k], new_cells[k+1] = new_cells[x0], new_cells[x1]
                new_cells[x0], new_cells[x1] = '.', '.'
                new_state = tuple(new_cells)
                if new_state not in visited:
                    visited.add(new_state)
                    # New empty cells are x and x+1
                    new_empty = (x-1, x)  # Adjust for zero-based indexing
                    queue.append((new_state, new_empty, operations + 1))
    
    # If target state is not reachable
    print(-1)

if __name__ == '__main__':
    main()