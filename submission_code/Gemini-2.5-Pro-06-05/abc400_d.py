# YOUR CODE HERE
import sys
from collections import deque

# It's a good practice for competitive programming in Python
# to potentially speed up I/O and increase recursion limit,
# although the latter is not needed for this iterative solution.
sys.setrecursionlimit(10**6)

def main():
    """
    Solves the problem using a 0-1 Breadth-First Search (BFS).
    The problem asks for the minimum number of "front kicks", which suggests
    a shortest path problem on a graph where edges have costs 0 or 1.
    
    - State: A cell (r, c) on the grid.
    - Cost: The minimum number of kicks to make that cell reachable.
    - 0-cost move: Walking to an adjacent road cell.
    - 1-cost move: Using a front kick. A kick from any reachable cell can
      clear walls up to 2 steps away in a cardinal direction. The effect of
      one kick from anywhere in a connected component of roads is equivalent
      to being able to reach any cell in a 5x5 area around any cell of that
      component.

    0-1 BFS is ideal for this scenario. We use a deque:
    - For 0-cost moves (walking), add the new cell to the front of the deque.
    - For 1-cost moves (kicking), add the new cell to the back of the deque.
    This ensures that we explore all possibilities at a certain kick-cost level
    before moving to the next level.
    """
    
    # Read problem parameters from standard input
    try:
        H, W = map(int, sys.stdin.readline().split())
        S = [sys.stdin.readline().strip() for _ in range(H)]
        A, B, C, D = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Graceful exit for certain testing environments
        return

    # Convert 1-based coordinates to 0-based for array indexing
    start_r, start_c = A - 1, B - 1
    end_r, end_c = C - 1, D - 1

    # dist[r][c] will store the minimum number of kicks to reach cell (r, c).
    # Initialize with -1 to indicate that a cell has not been visited yet.
    dist = [[-1] * W for _ in range(H)]

    # Use a deque for the 0-1 BFS algorithm.
    q = deque()

    # The starting cell is reachable with 0 kicks.
    dist[start_r][start_c] = 0
    q.append((start_r, start_c))

    # Perform the 0-1 BFS
    while q:
        r, c = q.popleft()
        k = dist[r][c]

        # --- Action 1: Move (Cost 0) ---
        # We can walk to adjacent cells that are roads in the original grid.
        # This connects components of original roads. If a new 'road' (created
        # by a kick) is adjacent to an original road, this connects them.
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < H and 0 <= nc < W and S[nr][nc] == '.' and dist[nr][nc] == -1:
                dist[nr][nc] = k
                q.appendleft((nr, nc))

        # --- Action 2: Front Kick (Cost 1) ---
        # A kick allows us to reach any cell in a 5x5 area around the current cell.
        # This is because from (r, c), we can walk to any adjacent reachable cell
        # and kick from there. The combined effect covers the 5x5 area.
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc

                if 0 <= nr < H and 0 <= nc < W and dist[nr][nc] == -1:
                    dist[nr][nc] = k + 1
                    q.append((nr, nc))
    
    # The answer is the minimum number of kicks to reach the destination cell.
    # The problem guarantees a path exists.
    print(dist[end_r][end_c])

if __name__ == '__main__':
    main()