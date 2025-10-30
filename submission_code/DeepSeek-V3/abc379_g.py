# YOUR CODE HERE
import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Precompute the positions of '?'
    q_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                q_positions.append((i, j))
    
    # Precompute the adjacency list for each cell
    adj = {}
    for i in range(H):
        for j in range(W):
            neighbors = []
            if i > 0:
                neighbors.append((i-1, j))
            if i < H-1:
                neighbors.append((i+1, j))
            if j > 0:
                neighbors.append((i, j-1))
            if j < W-1:
                neighbors.append((i, j+1))
            adj[(i, j)] = neighbors
    
    # Initialize the possible values for each cell
    possible = {}
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                possible[(i, j)] = {1, 2, 3}
            else:
                possible[(i, j)] = {int(grid[i][j])}
    
    # Check if the initial grid is already invalid
    for i in range(H):
        for j in range(W):
            for ni, nj in adj[(i, j)]:
                if grid[i][j] != '?' and grid[ni][nj] != '?' and grid[i][j] == grid[ni][nj]:
                    print(0)
                    return
    
    # Use BFS to propagate constraints
    queue = deque()
    for pos in q_positions:
        queue.append(pos)
    
    while queue:
        i, j = queue.popleft()
        current_possible = possible[(i, j)]
        for ni, nj in adj[(i, j)]:
            if (ni, nj) in possible:
                neighbor_possible = possible[(ni, nj)]
                # Remove any value from current_possible that is in neighbor_possible
                # Since they must be different
                new_possible = current_possible - neighbor_possible
                if new_possible != current_possible:
                    possible[(i, j)] = new_possible
                    if not new_possible:
                        print(0)
                        return
                    # If the possible set is reduced, re-add to queue
                    queue.append((i, j))
    
    # Now, count the number of valid assignments
    # Since the constraints are propagated, we can assign values in any order
    # The total number is the product of the sizes of the possible sets for each '?'
    total = 1
    for pos in q_positions:
        total *= len(possible[pos])
        total %= MOD
    
    print(total)

if __name__ == "__main__":
    main()