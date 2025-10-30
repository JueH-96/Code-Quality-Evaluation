import sys
from collections import deque

def solve():
    N, D = map(int, sys.stdin.readline().split())

    coords = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        coords.append((x, y))

    # Calculate D_squared to avoid square root operations and floating-point inaccuracies.
    # If distance <= D, then distance^2 <= D^2.
    D_squared = D * D

    # infected[i] will be True if person (i+1) is infected, False otherwise.
    # Initialize all people as not infected.
    infected = [False] * N

    # Person 1 (at index 0) is initially infected.
    infected[0] = True

    # Use a deque (double-ended queue) for Breadth-First Search (BFS).
    # This queue will store the indices of people who have recently become infected
    # and whose neighbors need to be checked for potential infection.
    q = deque()
    q.append(0) # Add person 1 (index 0) to the queue to start the spread.

    # BFS loop: continues as long as there are newly infected people to process.
    while q:
        u_idx = q.popleft() # Get an infected person from the queue.
        ux, uy = coords[u_idx] # Get their coordinates.

        # Iterate through all other people to see if they can be infected by u_idx.
        for v_idx in range(N):
            # If person v_idx is not already infected:
            if not infected[v_idx]:
                vx, vy = coords[v_idx] # Get their coordinates.

                # Calculate the squared Euclidean distance between person u_idx and person v_idx.
                dist_sq = (ux - vx)**2 + (uy - vy)**2

                # If the squared distance is less than or equal to D_squared,
                # person v_idx becomes infected.
                if dist_sq <= D_squared:
                    infected[v_idx] = True # Mark person v_idx as infected.
                    q.append(v_idx) # Add person v_idx to the queue to further spread the virus.

    # After the BFS loop completes, all possible infections have occurred.
    # Print the infection status for each person.
    for status in infected:
        if status:
            sys.stdout.write("Yes
")
        else:
            sys.stdout.write("No
")

# Call the solve function to run the program.
if __name__ == '__main__':
    solve()