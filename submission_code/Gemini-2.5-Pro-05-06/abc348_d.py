import sys
from collections import deque

# Read dimensions and grid
H, W = map(int, sys.stdin.readline().split())
grid_chars = [sys.stdin.readline().strip() for _ in range(H)]

# Read medicine data
N = int(sys.stdin.readline())
medicines_input_list = []
for i in range(N):
    r, c, e = map(int, sys.stdin.readline().split())
    # Store with 0-indexed coordinates and an original_id (0 to N-1)
    medicines_input_list.append({'original_id': i, 'r': r - 1, 'c': c - 1, 'e': e})

# Find S and T positions
start_pos = None
goal_pos = None
for r_idx in range(H):
    for c_idx in range(W):
        if grid_chars[r_idx][c_idx] == 'S':
            start_pos = (r_idx, c_idx)
        elif grid_chars[r_idx][c_idx] == 'T':
            goal_pos = (r_idx, c_idx)

# If S is T, goal is reached immediately
if start_pos[0] == goal_pos[0] and start_pos[1] == goal_pos[1]:
    sys.stdout.write("Yes
")
    sys.exit()

# Organize medicine data:
# med_at_coord: maps (r, c) tuple to medicine_object for quick lookup
# medicines_by_id: list of medicine_objects, indexed by original_id
med_at_coord = {}
medicines_by_id = [None] * N 
for med_data in medicines_input_list:
    med_at_coord[(med_data['r'], med_data['c'])] = med_data
    medicines_by_id[med_data['original_id']] = med_data

# BFS function to calculate shortest distances on the grid
def bfs_on_grid(s_r, s_c):
    q_grid = deque()
    q_grid.append((s_r, s_c, 0)) # (row, col, distance)
    
    # Initialize distances: -1 means not reachable or not visited yet
    dists = [[-1] * W for _ in range(H)]
    dists[s_r][s_c] = 0
    
    while q_grid:
        r, c, d = q_grid.popleft()

        # Explore neighbors: up, down, left, right
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            # Check bounds, not an obstacle, and not visited
            if 0 <= nr < H and 0 <= nc < W and \
               grid_chars[nr][nc] != '#' and dists[nr][nc] == -1:
                dists[nr][nc] = d + 1
                q_grid.append((nr, nc, d + 1))
    return dists

# Precompute distances from S to all cells
dists_from_S = bfs_on_grid(start_pos[0], start_pos[1])

# Precompute distances from each medicine's location to all cells
dists_from_meds_by_id = [None] * N 
for i in range(N): # i is the original_id of the medicine
    med = medicines_by_id[i]
    dists_from_meds_by_id[i] = bfs_on_grid(med['r'], med['c'])

# Takahashi must take a medicine at S to get initial energy
start_med_obj = med_at_coord.get(start_pos)
if start_med_obj is None: # No medicine at S
    sys.stdout.write("No
")
    sys.exit()

start_med_original_id = start_med_obj['original_id']
start_med_energy = start_med_obj['e']

# Check if T is reachable directly from S using start_med_energy
dist_S_to_T = dists_from_S[goal_pos[0]][goal_pos[1]]
if dist_S_to_T != -1 and start_med_energy >= dist_S_to_T:
    sys.stdout.write("Yes
")
    sys.exit()

# BFS on the "medicine graph"
q_med_bfs = deque() # Stores original_ids of medicines to visit
visited_meds = [False] * N # Tracks used medicines, indexed by original_id

# Mark the medicine at S as used
visited_meds[start_med_original_id] = True 

# Initialize queue with medicines reachable from S
for i in range(N): # i is original_id of a potential next medicine
    if visited_meds[i]: # Skip if already visited (e.g., if i is start_med_original_id)
         continue

    med_i_obj = medicines_by_id[i]
    # Distance from S to med_i_obj's location
    dist_S_to_med_i = dists_from_S[med_i_obj['r']][med_i_obj['c']]
    
    if dist_S_to_med_i != -1 and start_med_energy >= dist_S_to_med_i:
        q_med_bfs.append(i) # Add original_id to queue
        visited_meds[i] = True # Mark as enqueued/visited
    
# Process medicines from the queue
while q_med_bfs:
    curr_med_original_id = q_med_bfs.popleft()
    curr_med_obj = medicines_by_id[curr_med_original_id]
    curr_med_energy = curr_med_obj['e'] # Energy obtained from this medicine
    
    # Check if T is reachable from current medicine's location
    # Distances from curr_med_obj's location are precomputed
    dist_curr_to_T = dists_from_meds_by_id[curr_med_original_id][goal_pos[0]][goal_pos[1]]
    if dist_curr_to_T != -1 and curr_med_energy >= dist_curr_to_T:
        sys.stdout.write("Yes
")
        sys.exit()
        
    # Explore reaching other (unused) medicines
    for next_med_original_id in range(N):
        if visited_meds[next_med_original_id]: # If already used or enqueued
            continue
        
        next_med_obj = medicines_by_id[next_med_original_id]
        # Distance from curr_med_obj's location to next_med_obj's location
        dist_curr_to_next = dists_from_meds_by_id[curr_med_original_id][next_med_obj['r']][next_med_obj['c']]
        
        if dist_curr_to_next != -1 and curr_med_energy >= dist_curr_to_next:
            visited_meds[next_med_original_id] = True
            q_med_bfs.append(next_med_original_id)
            
# If BFS completes and T was not reached
sys.stdout.write("No
")