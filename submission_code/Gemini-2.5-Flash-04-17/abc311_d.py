import sys
from collections import deque

# Read input
N, M = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(N)]

# Use 0-based indexing internally
# Start position (2,2) in 1-based is (1,1) in 0-based
start_r, start_c = 1, 1

# Visited grid to track all touched ice squares
visited = [[False for _ in range(M)] for _ in range(N)]
# Enqueued grid to track positions added to queue (potential slide start points)
# This prevents processing the same starting point multiple times
enqueued = [[False for _ in range(M)] for _ in range(N)]

# Queue for BFS, stores positions where a new slide can start
q = deque()

# Start BFS
q.append((start_r, start_c))
# Mark the starting square as visited and enqueued
visited[start_r][start_c] = True
enqueued[start_r][start_c] = True

# Directions: Down, Up, Right, Left (dr, dc)
dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

while q:
    r, c = q.popleft() # This is a position where a slide starts

    # Try sliding in each direction
    for dr, dc in dirs:
        # Start the slide simulation from the current position (r, c)
        slide_r, slide_c = r, c

        # Calculate the next square in the chosen direction
        next_r, next_c = slide_r + dr, slide_c + dc

        # Simulate the slide: Keep moving as long as the *next* square is ice
        # The outer rock periphery guarantees that next_r, next_c will eventually hit '#'
        # The problem guarantees N, M >= 3 and outer periphery is '#', so (next_r, next_c)
        # will always be within grid bounds [0, N-1] x [0, M-1] when checked against grid,
        # provided (slide_r, slide_c) is within the inner ice region [1, N-2] x [1, M-2].
        # A slide stops before hitting rock, so (slide_r, slide_c) always remains within the inner region.
        while grid[next_r][next_c] == '.':
            # Move to the next square (it's ice)
            slide_r, slide_c = next_r, next_c

            # Mark this square as visited
            # The count is implicitly handled by checking the visited array later
            if not visited[slide_r][slide_c]:
                visited[slide_r][slide_c] = True

            # Calculate the next square to check from the new position
            next_r, next_c = slide_r + dr, slide_c + dc

        # The slide stops at (slide_r, slide_c), which is the last ice square before hitting rock.
        # This stopping position is a potential new starting point for slides.
        # Add this stopping position to the queue if it hasn't been enqueued before.
        if not enqueued[slide_r][slide_c]:
            enqueued[slide_r][slide_c] = True
            q.append((slide_r, slide_c))

# Count the number of visited ice squares
reachable_count = 0
for i in range(N):
    for j in range(M):
        if visited[i][j]:
            reachable_count += 1

# Print the result
print(reachable_count)