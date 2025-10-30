import heapq
import sys

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

# Read N
N = int(input())

# Read A, B, X for stages 1 to N-1.
# Store them in 0-based indexed lists A, B, X of size N-1.
# A[i], B[i], X[i] correspond to actions from stage i+1 (1-based)
A = [0] * (N - 1)
B = [0] * (N - 1)
X = [0] * (N - 1)
for i in range(N - 1):
    a, b, x = map(int, input().split())
    A[i] = a
    B[i] = b
    X[i] = x # X[i] is 1-based stage number

# dist[i] will store the minimum time to be able to play stage i (0-based index)
# Stages are 0, 1, ..., N-1.
dist = [float('inf')] * N # dist[i] = min time to reach stage i (0-based)
dist[0] = 0 # Stage 0 (corresponds to stage 1 in problem) takes 0 time to reach initially

# Priority queue: stores tuples (time, stage_index_0based)
# stage_index_0based ranges from 0 to N-1
pq = [(0, 0)] # Start at stage 0 (stage 1) with time 0

# Dijkstra's algorithm
while pq:
    current_time, u = heapq.heappop(pq) # u is 0-based stage index

    # If we already found a shorter path to stage u, skip this one
    if current_time > dist[u]:
        continue

    # If we reached the target stage N-1 (stage N in problem), we have found the minimum time
    # The question asks for the minimum time to *be able to play* stage N,
    # which corresponds to reaching stage N-1 in 0-based indexing.
    # dist[N-1] holds this value. When (dist[N-1], N-1) is popped,
    # dist[N-1] is finalized as the minimum time.
    if u == N - 1:
        print(dist[N - 1])
        break

    # Consider actions from stage u (0-based).
    # Actions A[u], B[u], X[u] are defined for stage u+1 (1-based).
    # These actions are available only if u+1 <= N-1, i.e., u <= N-2.
    # The loop processes u from 0 up to N-1. Actions are only possible
    # if we are at stage u where u < N-1.
    # When u reaches N-1, the loop breaks before considering actions.
    # So, accessing A[u], B[u], X[u] where u is in [0, N-2] is always safe.

    # Action 1: clear stage u+1 (1-based) with cost A[u], unlock stage u+2 (1-based)
    # Target stage is u+1 (0-based index).
    neighbor_index_0based_A = u + 1
    cost_A = A[u]
    new_time_A = current_time + cost_A
    if new_time_A < dist[neighbor_index_0based_A]:
        dist[neighbor_index_0based_A] = new_time_A
        heapq.heappush(pq, (new_time_A, neighbor_index_0based_A))

    # Action 2: clear stage u+1 (1-based) with cost B[u], unlock stage X[u] (1-based)
    # Target stage is X[u] (1-based). Convert to 0-based index.
    neighbor_index_0based_B = X[u] - 1
    cost_B = B[u]
    new_time_B = current_time + cost_B
    if new_time_B < dist[neighbor_index_0based_B]:
        dist[neighbor_index_0based_B] = new_time_B
        heapq.heappush(pq, (new_time_B, neighbor_index_0based_B))

# If the loop finishes without breaking (e.g., target is unreachable - not possible here
# since N>=2 and path i -> i+1 always exists) or pq is empty.
# The break statement ensures we exit as soon as the shortest path to N-1 is found.