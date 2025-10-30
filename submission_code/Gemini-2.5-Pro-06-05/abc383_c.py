# YOUR CODE HERE
import sys
from collections import deque

def main():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        # Read problem parameters
        H, W, D = map(int, input().split())
        S = [input().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors gracefully
        print(0)
        return

    # is_humidified[r][c] will be True if the cell (r,c) is humidified, False otherwise.
    is_humidified = [[False] * W for _ in range(H)]
    
    # q_main is the queue for the main BFS on the implicit propagation graph G'.
    # It stores cells that have been humidified and whose neighbors in G' need to be explored.
    q_main = deque()

    # Initialize the queue with all humidifier locations.
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'H':
                # Avoid adding duplicate humidifiers if they are already marked
                if not is_humidified[r][c]:
                    is_humidified[r][c] = True
                    q_main.append((r, c))

    # This is the main BFS that simulates the propagation of humidification.
    while q_main:
        r_start, c_start = q_main.popleft()

        # For each humidified cell, we start an "inner" BFS to find all other
        # floor cells within distance D. These are its neighbors in G'.
        q_inner = deque([(r_start, c_start, 0)])
        visited_inner = {(r_start, c_start)}

        while q_inner:
            r, c, d = q_inner.popleft()

            # If we have reached the maximum propagation distance D, stop exploring from this path.
            if d >= D:
                continue

            # Explore neighbors in the grid (up, down, left, right).
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                # Check if the neighbor is within bounds, not a wall, and not yet visited in this inner BFS.
                if 0 <= nr < H and 0 <= nc < W and S[nr][nc] != '#' and (nr, nc) not in visited_inner:
                    visited_inner.add((nr, nc))
                    
                    # If this neighbor cell has not been humidified yet, mark it as humidified
                    # and add it to the main queue to propagate from it later.
                    if not is_humidified[nr][nc]:
                        is_humidified[nr][nc] = True
                        q_main.append((nr, nc))
                    
                    # Continue the inner BFS from this new cell.
                    q_inner.append((nr, nc, d + 1))
    
    # After the propagation is complete, count all cells marked as humidified.
    count = 0
    for r in range(H):
        for c in range(W):
            if is_humidified[r][c]:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()