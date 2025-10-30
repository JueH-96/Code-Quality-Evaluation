import collections
import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Read the grid
    # grid[i][j] corresponds to the square at row i+1, column j+1 (1-indexed)
    # Using 0-indexed coordinates internally, so (1,1) in problem is grid[0][0], etc.
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Starting position (2,2) in 1-indexed, which is (1,1) in 0-indexed
    start_r, start_c = 1, 1

    # Queue for BFS, storing (row, col) of reachable *stopping* points.
    # These are the points from which new slides can originate.
    q = collections.deque()
    q.append((start_r, start_c))

    # Set to keep track of squares from which we have already started a slide.
    # This prevents redundant exploration from the same stopping point (node in graph).
    visited_reachable_squares = set()
    visited_reachable_squares.add((start_r, start_c))

    # Set to keep track of all unique ice squares touched (passed through or stopped on).
    touched_ice_squares = set()
    touched_ice_squares.add((start_r, start_c)) # The starting square (2,2) is always touched.

    # Directions: (dr, dc) for Up, Down, Left, Right
    # dr: change in row, dc: change in column
    directions = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]

    while q:
        r, c = q.popleft() # Current position, from which a new slide can start

        # Try sliding in all four directions from (r, c)
        for dr, dc in directions:
            current_slide_r, current_slide_c = r, c # This tracks the player's position *during* the slide

            while True:
                # Calculate the next square in the current slide direction
                next_r, next_c = current_slide_r + dr, current_slide_c + dc

                # Check if the next square is rock
                # The problem states the outer periphery is rock, so (next_r, next_c) will always be valid grid indices.
                if grid[next_r][next_c] == '#':
                    # The player stops at (current_slide_r, current_slide_c) because (next_r, next_c) is rock.
                    # This (current_slide_r, current_slide_c) is a reachable *stopping* point.
                    # If this stopping point hasn't been processed as a starting point for slides,
                    # add it to the queue to explore moves from it.
                    if (current_slide_r, current_slide_c) not in visited_reachable_squares:
                        visited_reachable_squares.add((current_slide_r, current_slide_c))
                        q.append((current_slide_r, current_slide_c))
                    break # Slide ends because a rock was hit

                else: # grid[next_r][next_c] == '.' (ice)
                    # The player moves to (next_r, next_c). This square is passed through.
                    # Add it to the set of all unique touched ice squares.
                    touched_ice_squares.add((next_r, next_c))
                    # Continue sliding from this new position
                    current_slide_r, current_slide_c = next_r, next_c

    # The final answer is the total number of unique ice squares touched.
    print(len(touched_ice_squares))

# Call the solve function to run the program
solve()