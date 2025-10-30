import sys
from collections import deque

def bfs(N, start_state, target_state):
    if start_state == target_state:
        return 0
    queue = deque()
    visited = set()
    queue.append((start_state, 0))
    visited.add(start_state)
    while queue:
        curr, steps = queue.popleft()
        # Find empty k
        for i in range(N + 1):  # k from 0 to N
            if curr[i] == '.' and curr[i + 1] == '.':
                empty_k = i
                break
        # Find all possible x_idx
        for x_idx in range(0, N + 1):  # x_idx from 0 to N
            if curr[x_idx] != '.' and curr[x_idx + 1] != '.':
                # Create new state
                new_state_list = list(curr)
                new_state_list[empty_k] = curr[x_idx]
                new_state_list[empty_k + 1] = curr[x_idx + 1]
                new_state_list[x_idx] = '.'
                new_state_list[x_idx + 1] = '.'
                new_state = ''.join(new_state_list)
                if new_state == target_state:
                    return steps + 1
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
    return -1

# Read input
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
T = data[2]

# Check if color counts match
if S.count('B') != T.count('B'):
    print(-1)
    sys.exit()

# Create start and target states
start_state = S + ".."
target_state = T + ".."

# Run BFS
dist = bfs(N, start_state, target_state)
print(dist)