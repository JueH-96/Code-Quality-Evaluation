import collections
import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Player starts at (2,2), which is 1-indexed.
    start_r, start_c = 2, 2

    q = collections.deque()
    
    # Set to store all unique ice squares touched (passed or rested on)
    touched_squares = set()
    
    # Set to store resting points that have been enqueued or already processed.
    # This prevents adding the same resting point multiple times to the queue
    # and thus avoids redundant processing.
    enqueued_or_processed_resting_points = set()

    # Initial state: player is at (start_r, start_c)
    q.append((start_r, start_c))
    touched_squares.add((start_r, start_c)) # Starting square is touched
    enqueued_or_processed_resting_points.add((start_r, start_c))

    # Directions: Right, Left, Down, Up
    # (dr, dc) pairs for (row_change, col_change)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    while q:
        curr_r, curr_c = q.popleft() # Current resting square to explore from

        for dr, dc in directions:
            # path_r, path_c track the player's current position during a single slide.
            # It starts at the square from which the move is initiated (curr_r, curr_c).
            path_r, path_c = curr_r, curr_c
            
            while True:
                # next_try_r, next_try_c are coordinates of the square player attempts to move into.
                next_try_r, next_try_c = path_r + dr, path_c + dc
                
                # Grid access is 0-indexed: grid[row_idx][col_idx]
                # Player coordinates (r,c) are 1-indexed. So convert: grid[r-1][c-1].
                # The problem guarantees periphery is rock, so player will hit a rock
                # at grid boundary if they move towards it. No explicit boundary check needed.
                if grid[next_try_r-1][next_try_c-1] == '.':
                    # If the next square is ice, player moves there. This square is touched.
                    path_r, path_c = next_try_r, next_try_c # Update player's position
                    touched_squares.add((path_r, path_c))   # Mark as touched
                else:
                    # If the next square is rock, player cannot move there.
                    # Player stops at the current (path_r, path_c).
                    break 
            
            # After the slide, player rests at (path_r, path_c).
            # If this resting point has not been enqueued or processed before,
            # add it to the queue and mark it as such.
            if (path_r, path_c) not in enqueued_or_processed_resting_points:
                enqueued_or_processed_resting_points.add((path_r, path_c))
                q.append((path_r, path_c))
                
    sys.stdout.write(str(len(touched_squares)) + "
")

if __name__ == '__main__':
    main()