import sys

# Increase recursion depth for DFS
# Maximum number of vertices is 2 * 200000 = 400000
# A path could potentially visit all vertices. Set limit > 400000.
sys.setrecursionlimit(500000)

def add_edge(u, v):
    graph[u].append(v)
    rev_graph[v].append(u)

def dfs1(u):
    visited[u] = True
    for v in graph[u]:
        if not visited[v]:
            dfs1(v)
    order.append(u)

def dfs2(u, component):
    visited[u] = True
    scc[u] = component
    for v in rev_graph[u]:
        if not visited[v]:
            dfs2(v, component)

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# If A contains no 0s, answer is Yes.
if all(a == 1 for a in A):
    print("Yes")
    sys.exit()

# 2N variables: P_0..P_{N-1}, !P_0..!P_{N-1}
# P_i is true if index i can be the start of an 'ARC' or 'CRA' pattern in S.
# Map P_i to index i (0 to N-1), !P_i to index i+N (N to 2N-1)
V = 2 * N
graph = [[] for _ in range(V)]
rev_graph = [[] for _ in range(V)]

# Add edges from general constraints on P_i:
# If index i is a start position, indices i+1 and i+2 (circularly) cannot be start positions.
# P_i => !P_{i+1} and P_i => !P_{i+2} (circularly)
# Equivalent implications for 2-SAT:
# !P_i or !P_{i+1} <=> P_i => !P_{i+1}, P_{i+1} => !P_i
# !P_i or !P_{i+2} <=> P_i => !P_{i+2}, P_{i+2} => !P_i
for i in range(N):
    # P_i => !P_{(i+1)%N}
    u = i # P_i
    v = (i + 1) % N + N # !P_{(i+1)%N}
    add_edge(u, v)
    # P_{(i+1)%N} => !P_i
    u_rev = (i + 1) % N # P_{(i+1)%N}
    v_rev = i + N # !P_i
    add_edge(u_rev, v_rev)

    # P_i => !P_{(i+2)%N}
    u = i # P_i
    v = (i + 2) % N + N # !P_{(i+2)%N}
    add_edge(u, v)
    # P_{(i+2)%N} => !P_i
    u_rev = (i + 2) % N # P_{(i+2)%N}
    v_rev = i + N # !P_i
    add_edge(u_rev, v_rev)

# Add edges from specific constraints (A_j=0):
# For every index j where A_j=0, it must be possible to cover it.
# An operation at index i covers A_i and A_{i+1}.
# A_j must be covered by an operation starting at j (covering A_j, A_{j+1})
# OR by an operation starting at j-1 (covering A_{j-1}, A_j).
# This means index j must be a start position (P_j is true) OR index j-1 must be a start position (P_{j-1} is true).
# P_j or P_{(j-1)%N} (circularly)
# Equivalent implications for 2-SAT:
# !P_j => P_{(j-1)%N}
# !P_{(j-1)%N} => P_j
for j in range(N):
    if A[j] == 0:
        prev_j = (j - 1 + N) % N
        # !P_j => P_{prev_j}
        u = j + N # !P_j
        v = prev_j # P_{prev_j}
        add_edge(u, v)
        # !P_{prev_j} => P_j
        u_rev = prev_j + N # !P_{prev_j}
        v_rev = j # P_j
        add_edge(u_rev, v_rev)

# Kosaraju's algorithm
visited = [False] * V
order = []
# First DFS to get finishing times
for i in range(V):
    if not visited[i]:
        dfs1(i)

visited = [False] * V
scc = [-1] * V # Stores the SCC component id for each vertex
component = 0
# Second DFS on reversed graph in reverse finishing order
# Process nodes in decreasing order of finishing times
for u in reversed(order):
    if not visited[u]:
        dfs2(u, component)
        component += 1

# Check for contradiction: P_i and !P_i in the same SCC
# P_i is vertex i, !P_i is vertex i+N
has_solution = True
for i in range(N):
    if scc[i] == scc[i + N]:
        has_solution = False
        break

if has_solution:
    print("Yes")
else:
    print("No")