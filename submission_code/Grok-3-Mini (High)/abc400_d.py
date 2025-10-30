import sys
import collections

# Read all input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
S = []
for _ in range(H):
    s_str = data[index]
    index += 1
    S.append(s_str)
A = int(data[index])
index += 1
B = int(data[index])
index += 1
C = int(data[index])
index += 1
D = int(data[index])
index += 1  # Move index forward, though not necessary after last read

# Convert to 0-based indexing
start_i = A - 1
start_j = B - 1
end_i = C - 1
end_j = D - 1

INF = 2000005  # A large number larger than possible maximum kicks
dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

# 0-1 BFS setup
start_added = frozenset()
start_state = (start_i, start_j, start_added)
dist = {}
dist[start_state] = 0
queue = collections.deque([start_state])

while queue:
    current_state = queue.popleft()
    cur_i, cur_j, cur_added = current_state
    cur_kicks = dist[current_state]
    
    # Check if reached the end
    if cur_i == end_i and cur_j == end_j:
        print(cur_kicks)
        sys.exit()
    
    # Generate move neighbors (cost 0)
    for di, dj in dirs:
        ni = cur_i + di
        nj = cur_j + dj
        if 0 <= ni < H and 0 <= nj < W:
            if S[ni][nj] == '.' or (ni, nj) in cur_added:  # Cell is road
                new_state = (ni, nj, cur_added)
                new_cost = cur_kicks
                if new_cost < dist.get(new_state, INF):
                    dist[new_state] = new_cost
                    queue.appendleft(new_state)  # Add to front for cost 0
    
    # Generate kick neighbors (cost 1)
    for delta in dirs:
        di, dj = delta
        cells_to_add = set()
        # Check cell at distance 1
        ni1 = cur_i + di
        nj1 = cur_j + dj
        if 0 <= ni1 < H and 0 <= nj1 < W and S[ni1][nj1] == '#' and (ni1, nj1) not in cur_added:
            cells_to_add.add((ni1, nj1))
        # Check cell at distance 2
        ni2 = cur_i + 2 * di
        nj2 = cur_j + 2 * dj
        if 0 <= ni2 < H and 0 <= nj2 < W and S[ni2][nj2] == '#' and (ni2, nj2) not in cur_added:
            cells_to_add.add((ni2, nj2))
        # If there are cells to add, create new state
        if cells_to_add:
            new_added_elements = set(cur_added) | cells_to_add
            new_added_frozenset = frozenset(new_added_elements)
            new_state = (cur_i, cur_j, new_added_frozenset)  # Same position
            new_cost = cur_kicks + 1
            if new_cost < dist.get(new_state, INF):
                dist[new_state] = new_cost
                queue.append(new_state)  # Add to back for cost 1

# If no path found
print(-1)