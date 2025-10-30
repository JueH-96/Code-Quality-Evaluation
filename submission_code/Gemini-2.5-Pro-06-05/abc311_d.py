import sys
import collections

def main():
    """
    This function reads input from stdin, solves the problem, and prints the answer to stdout.
    The problem is modeled as a graph traversal on the grid. We use Breadth-First Search (BFS)
    to find all reachable squares.
    """
    # Read grid dimensions
    try:
        N, M = map(int, sys.stdin.readline().split())
    except ValueError:
        # Handles empty input case
        return

    # Read the grid layout
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # The problem uses 1-based indexing for coordinates. We convert to 0-based.
    # The starting position (2,2) becomes (1,1).
    start_pos = (1, 1)

    # A set to store all squares that have been touched (passed through or rested on).
    # We initialize it with the start position.
    touched = {start_pos}
    
    # A queue for BFS. It stores coordinates of resting spots to be explored.
    q = collections.deque([start_pos])
    
    # A set to keep track of resting spots already added to the queue to avoid redundant work.
    visited_resting_spots = {start_pos}

    # Define the four directions of movement: Right, Left, Down, Up.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Perform BFS to explore all reachable resting spots.
    while q:
        r, c = q.popleft()

        # From the current resting spot, try moving in each of the four directions.
        for dr, dc in directions:
            # path_r and path_c will trace the slide.
            path_r, path_c = r, c
            
            # The grid is guaranteed to have a rock border, so we don't need
            # to check for out-of-bounds access during the slide.
            while S[path_r + dr][path_c + dc] == '.':
                path_r += dr
                path_c += dc
                touched.add((path_r, path_c))
            
            # The slide ends at new_resting_spot.
            new_resting_spot = (path_r, path_c)
            
            # If this resting spot has not been explored from yet, add it to the queue.
            if new_resting_spot not in visited_resting_spots:
                visited_resting_spots.add(new_resting_spot)
                q.append(new_resting_spot)
    
    # The final answer is the total number of unique squares touched.
    print(len(touched))

if __name__ == "__main__":
    main()