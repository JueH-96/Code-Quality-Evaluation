import collections
import sys

def solve():
    H, W, D = map(int, sys.stdin.readline().split())
    S_grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize distances to infinity
    # float('inf') is a common way to represent infinity
    distances = [[float('inf')] * W for _ in range(H)]
    
    # Queue for BFS: stores (row, col) tuples
    queue = collections.deque()

    # Find all humidifier locations, set their distance to 0, and add to queue
    for r_idx in range(H):
        for c_idx in range(W):
            if S_grid[r_idx][c_idx] == 'H':
                distances[r_idx][c_idx] = 0
                queue.append((r_idx, c_idx))

    # Delta row and delta column for neighbors (up, down, left, right)
    delta_r = [-1, 1, 0, 0]
    delta_c = [0, 0, -1, 1]
    
    # BFS
    while queue: # Loop as long as there are cells to process
        curr_r, curr_c = queue.popleft()
        curr_dist = distances[curr_r][curr_c]

        # Optimization: if current distance is D, its neighbors will be at distance D+1.
        # These neighbors cannot be humidified within D moves via this path.
        # Since BFS finds shortest paths, any path to them will be at least D+1.
        # So, no need to explore further from this cell if its distance is already D.
        if curr_dist == D:
            continue

        for i in range(4): # For each of the 4 directions
            next_r, next_c = curr_r + delta_r[i], curr_c + delta_c[i]

            # Check if the neighbor is within grid boundaries
            if 0 <= next_r < H and 0 <= next_c < W:
                # Check if the neighbor is not a wall and has not been visited yet
                # (distances[next_r][next_c] == float('inf') means unvisited in unweighted BFS)
                if S_grid[next_r][next_c] != '#' and distances[next_r][next_c] == float('inf'):
                    distances[next_r][next_c] = curr_dist + 1
                    queue.append((next_r, next_c))
    
    # Count humidified floor cells
    humidified_count = 0
    for r_idx in range(H):
        for c_idx in range(W):
            # A cell is a floor cell if it's not a wall (original S was '.' or 'H')
            if S_grid[r_idx][c_idx] != '#':
                # Check if it's humidified (reachable within D moves)
                # distances[r_idx][c_idx] could be float('inf') if unreachable
                if distances[r_idx][c_idx] <= D: 
                    humidified_count += 1
    
    sys.stdout.write(str(humidified_count) + "
")

if __name__ == '__main__':
    solve()