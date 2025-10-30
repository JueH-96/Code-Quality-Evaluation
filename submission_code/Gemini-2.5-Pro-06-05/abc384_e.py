# YOUR CODE HERE
import sys
import heapq

def solve():
    """
    Reads input, solves the slime absorption problem, and prints the result.
    """
    # Use fast I/O
    readline = sys.stdin.readline

    # Read problem parameters
    H, W, X = map(int, readline().split())
    P, Q = map(int, readline().split())
    
    # Adjust P, Q to be 0-indexed for list access
    P -= 1
    Q -= 1
    
    # Read the grid of slime strengths
    S = [list(map(int, readline().split())) for _ in range(H)]

    # The problem is solved with a greedy approach similar to Prim's algorithm.
    # We always try to absorb the weakest available slime on the frontier. This
    # is optimal because it minimally increases Takahashi's strength, maximizing
    # the chances of being able to absorb other, stronger slimes later.

    # Initialize Takahashi's strength with the starting slime.
    current_strength = S[P][Q]

    # Priority queue (min-heap) to store slimes on the frontier.
    # Each element is a tuple: (strength, row, col).
    frontier_pq = []

    # A set to track cells that are either part of Takahashi's territory
    # or are already in the frontier_pq. This prevents processing the same slime twice.
    visited = set()
    visited.add((P, Q))

    def add_neighbors_to_frontier(r, c):
        """Adds unvisited neighbors of cell (r, c) to the frontier priority queue."""
        # Directions for adjacent cells: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            # Check if the neighbor is within the grid boundaries and has not been visited
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                # Add the neighbor to the priority queue and mark it as visited
                heapq.heappush(frontier_pq, (S[nr][nc], nr, nc))
                visited.add((nr, nc))

    # Initialize the frontier with neighbors of the starting cell.
    add_neighbors_to_frontier(P, Q)

    # Greedily absorb slimes from the frontier.
    while frontier_pq:
        # Get the weakest slime from the frontier.
        slime_strength, r, c = heapq.heappop(frontier_pq)

        # Check the absorption condition: strength < T/X  <=> strength * X < T.
        # Using multiplication avoids potential floating-point precision issues.
        if slime_strength * X < current_strength:
            # Absorb the slime: update strength and expand territory.
            current_strength += slime_strength
            
            # Add neighbors of the newly absorbed slime to the frontier.
            add_neighbors_to_frontier(r, c)
        else:
            # If the weakest slime cannot be absorbed, no other slime in the
            # frontier can be absorbed either, as they are all stronger.
            # The process terminates.
            break

    # Print the final maximum strength achieved.
    print(current_strength)

# Run the solution
solve()