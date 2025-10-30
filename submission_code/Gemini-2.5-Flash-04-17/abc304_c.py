import sys
import collections

# Read N and D
line = sys.stdin.readline().split()
N = int(line[0])
D = int(line[1])

# Calculate D squared to avoid using square root
D_squared = D * D

# Read coordinates
coords = []
for _ in range(N):
    line = sys.stdin.readline().split()
    x = int(line[0])
    y = int(line[1])
    coords.append((x, y))

# Initialize infection status
# infected[i] corresponds to person i+1 (with 0-based indexing)
infected = [False] * N
infected[0] = True # Person 1 (index 0) is initially infected

# BFS queue
q = collections.deque()
q.append(0) # Add index of person 1 (0) to the queue

# BFS loop
while q:
    u = q.popleft() # Get the index of an infected person (u corresponds to person u+1)

    # Check all other people v (v corresponds to person v+1)
    for v in range(N):
        # Skip if same person or already infected
        if v == u or infected[v]:
            continue

        # Calculate squared distance between person u and person v
        ux, uy = coords[u]
        vx, vy = coords[v]
        dx = ux - vx
        dy = uy - vy
        dist_squared = dx*dx + dy*dy

        # If distance is within D, infect person v
        if dist_squared <= D_squared:
            infected[v] = True
            q.append(v) # Add the newly infected person to the queue

# Print results
for is_infected in infected:
    if is_infected:
        print("Yes")
    else:
        print("No")