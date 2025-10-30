import sys
from collections import deque

def main():
    """
    Solves the virus spread problem by modeling it as finding a connected
    component in a graph using Breadth-First Search (BFS).
    """
    # Read problem parameters N (number of people) and D (spread distance)
    try:
        N, D = map(int, sys.stdin.readline().split())
    except (ValueError, IndexError):
        # Handle potential empty input
        return

    # Calculate D squared for efficient distance comparison (avoids sqrt)
    D_squared = D * D

    # Read the coordinates of N people into a list of tuples
    coords = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # `infected` list tracks the state of each person.
    # It also serves as the 'visited' set for the Breadth-First Search (BFS).
    # infected[i] corresponds to person i+1.
    infected = [False] * N

    # Queue for BFS, storing 0-based indices of people to visit
    q = deque()

    # The problem states person 1 is initially infected. We use 0-based indexing.
    infected[0] = True
    q.append(0)

    # Perform BFS to find all people reachable from person 1
    while q:
        # Get the next infected person from the queue
        current_idx = q.popleft()
        x1, y1 = coords[current_idx]

        # Check for potential spread to all other people
        for next_idx in range(N):
            # If the person is not already infected, check the distance
            if not infected[next_idx]:
                x2, y2 = coords[next_idx]
                
                # Calculate squared Euclidean distance to avoid floating-point math
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                
                # If within spread distance, the virus spreads
                if dist_sq <= D_squared:
                    infected[next_idx] = True
                    q.append(next_idx)

    # Output the final infection status for each person
    for is_person_infected in infected:
        print("Yes" if is_person_infected else "No")

if __name__ == "__main__":
    main()