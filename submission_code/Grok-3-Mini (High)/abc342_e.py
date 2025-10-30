import sys
import heapq

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Build the incoming trains list for each station
incoming = [[] for _ in range(N + 1)]
for _ in range(M):
    l_val = int(data[index])
    index += 1
    d_val = int(data[index])
    index += 1
    k_val = int(data[index])
    index += 1
    c_val = int(data[index])
    index += 1
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    # For train from A to B, add to incoming of B
    incoming[B].append((A, l_val, d_val, k_val, c_val))

# Define INF and MINUS_INF
INF = 2000000000000000000  # 2e18
MINUS_INF = -2000000000000000000

# Initialize g values
g = [MINUS_INF] * (N + 1)
g[N] = INF

# Priority queue for Dijkstra (min-heap on negative g)
heap = []
heapq.heappush(heap, (-INF, N))  # (negative g, node)

# Dijkstra's algorithm
while heap:
    neg_g_u, u = heapq.heappop(heap)
    g_enqueued = -neg_g_u
    
    # Skip if this entry is outdated (g[u] has been increased)
    if g_enqueued < g[u]:
        continue
    
    # For each train arriving at u (incoming edges)
    for train in incoming[u]:
        W, l, d, k, c = train
        
        # Compute X = min(t_max, g[u] - c) or handle INF
        if g[u] == INF:
            X = l + (k - 1) * d  # t_max
        else:
            X = min(l + (k - 1) * d, g[u] - c)
        
        # Check if there is a valid departure
        if X >= l:
            m_max = min(k - 1, (X - l) // d)  # floor division
            g_W_candidate = l + m_max * d
            
            # Update g[W] if better
            if g_W_candidate > g[W]:
                g[W] = g_W_candidate
                heapq.heappush(heap, (-g_W_candidate, W))

# Output the results for stations 1 to N-1
for S in range(1, N):  # S from 1 to N-1
    if g[S] > MINUS_INF:
        print(g[S])
    else:
        print("Unreachable")