def solve(H, W, grid):
    MOD = 998244353
    
    # Define states for tiles in different orientations
    # Each state is (t, r, b, l) representing connections at top, right, bottom, left edges
    a_states = [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 0, 1)]  # Type A
    b_states = [(1, 0, 1, 0), (0, 1, 0, 1)]  # Type B
    
    # Initialize the state of each edge in the grid
    # -1 means undetermined, 0 means no line segment, 1 means line segment
    edge_state = {}
    for i in range(H):
        for j in range(W):
            for e in range(4):  # 0: top, 1: right, 2: bottom, 3: left
                edge_state[(i, j, e)] = -1
    
    # Memoization
    memo = {}
    
    # Check if the current state is consistent
    def is_consistent(i, j, t, r, b, l):
        # Check if the top edge is consistent
        if edge_state[(i, j, 0)] != -1 and edge_state[(i, j, 0)] != t:
            return False
        # Check if the right edge is consistent
        if edge_state[(i, j, 1)] != -1 and edge_state[(i, j, 1)] != r:
            return False
        # Check if the bottom edge is consistent
        if edge_state[(i, j, 2)] != -1 and edge_state[(i, j, 2)] != b:
            return False
        # Check if the left edge is consistent
        if edge_state[(i, j, 3)] != -1 and edge_state[(i, j, 3)] != l:
            return False
        return True
    
    # Define a function to update the edge state
    def update_edge_state(i, j, t, r, b, l):
        # Update the edge state for the current cell
        edge_state[(i, j, 0)] = t
        edge_state[(i, j, 1)] = r
        edge_state[(i, j, 2)] = b
        edge_state[(i, j, 3)] = l
        
        # Update the edge state for the neighboring cells
        # Top neighbor, wrap around if needed
        if i > 0:
            edge_state[(i-1, j, 2)] = t
        else:  # Wrap around to the bottom
            edge_state[(H-1, j, 2)] = t
        
        # Right neighbor, wrap around if needed
        if j < W-1:
            edge_state[(i, j+1, 3)] = r
        else:  # Wrap around to the leftmost column
            edge_state[(i, 0, 3)] = r
        
        # Bottom neighbor, wrap around if needed
        if i < H-1:
            edge_state[(i+1, j, 0)] = b
        else:  # Wrap around to the top
            edge_state[(0, j, 0)] = b
        
        # Left neighbor, wrap around if needed
        if j > 0:
            edge_state[(i, j-1, 1)] = l
        else:  # Wrap around to the rightmost column
            edge_state[(i, W-1, 1)] = l
    
    # Define a function to rollback the edge state
    def rollback_edge_state(i, j):
        # Rollback the edge state for the current cell
        edge_state[(i, j, 0)] = -1
        edge_state[(i, j, 1)] = -1
        edge_state[(i, j, 2)] = -1
        edge_state[(i, j, 3)] = -1
        
        # Rollback the edge state for the neighboring cells
        # Top neighbor, wrap around if needed
        if i > 0:
            edge_state[(i-1, j, 2)] = -1
        else:  # Wrap around to the bottom
            edge_state[(H-1, j, 2)] = -1
        
        # Right neighbor, wrap around if needed
        if j < W-1:
            edge_state[(i, j+1, 3)] = -1
        else:  # Wrap around to the leftmost column
            edge_state[(i, 0, 3)] = -1
        
        # Bottom neighbor, wrap around if needed
        if i < H-1:
            edge_state[(i+1, j, 0)] = -1
        else:  # Wrap around to the top
            edge_state[(0, j, 0)] = -1
        
        # Left neighbor, wrap around if needed
        if j > 0:
            edge_state[(i, j-1, 1)] = -1
        else:  # Wrap around to the rightmost column
            edge_state[(i, W-1, 1)] = -1
    
    # Recursively count valid arrangements
    def backtrack(i, j):
        if i == H:  # All cells processed
            return 1
        
        if j == W:  # Move to next row
            return backtrack(i+1, 0)
        
        # Get the key for memoization
        key = (i, j)
        # Only include non-undetermined edge states in the key
        for ni in range(H):
            for nj in range(W):
                for e in range(4):
                    if edge_state[(ni, nj, e)] != -1:
                        key = key + ((ni, nj, e, edge_state[(ni, nj, e)]),)
        
        # Check if this state has already been computed
        if key in memo:
            return memo[key]
        
        count = 0
        tile_type = grid[i][j]
        if tile_type == 'A':
            for t, r, b, l in a_states:
                if is_consistent(i, j, t, r, b, l):
                    update_edge_state(i, j, t, r, b, l)
                    count = (count + backtrack(i, j+1)) % MOD
                    rollback_edge_state(i, j)
        else:  # Type B
            for t, r, b, l in b_states:
                if is_consistent(i, j, t, r, b, l):
                    update_edge_state(i, j, t, r, b, l)
                    count = (count + backtrack(i, j+1)) % MOD
                    rollback_edge_state(i, j)
        
        # Memoize the result
        memo[key] = count
        return count
    
    return backtrack(0, 0)

def main():
    T = int(input())
    for _ in range(T):
        H, W = map(int, input().split())
        grid = []
        for _ in range(H):
            grid.append(input())
        print(solve(H, W, grid))

if __name__ == "__main__":
    main()