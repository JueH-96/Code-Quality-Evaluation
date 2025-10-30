from collections import deque

def min_stairs(H, W, F, A, B, Y, C, D, Z):
    # Initialize a queue for BFS
    queue = deque([(A, B, Y)])
    # Initialize a set to keep track of visited cells
    visited = set([(A, B, Y)])
    # Initialize the number of stairs used
    stairs = 0

    while queue:
        # Get the current position
        i, j, x = queue.popleft()

        # If we've reached the target floor, return the number of stairs used
        if i == C and j == D and x == Z:
            return stairs

        # Try moving up or down one floor within the same building
        if x > 1:
            queue.append((i, j, x - 1))
            visited.add((i, j, x - 1))
        if x < F[i-1][j-1]:
            queue.append((i, j, x + 1))
            visited.add((i, j, x + 1))

        # Try moving to adjacent buildings
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 1 <= ni <= H and 1 <= nj <= W and F[ni-1][nj-1] >= x:
                queue.append((ni, nj, x))
                visited.add((ni, nj, x))

        # Increment the number of stairs used
        stairs += 1

    # If we couldn't reach the target floor, return -1
    return -1

# Read the input
H, W = map(int, input().split())
F = [list(map(int, input().split())) for _ in range(H)]
Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]

# Solve each query and print the result
for A, B, Y, C, D, Z in queries:
    print(min_stairs(H, W, F, A, B, Y, C, D, Z))