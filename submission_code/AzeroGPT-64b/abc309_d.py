from collections import Counter
from collections import deque

def bfs(edges, start, N):
  D = [-1] * N
  D[start] = 0
  queue = deque([start])
  while queue:
    current = queue.popleft()
    for next in edges[current]:
      if D[next] == -1:
        D[next] = D[current] + 1
        queue.append(next)
  return D

def solve(N1, N2, M, edges):
  N = N1+N2

  c = Counter()
  for a, b in edges:
    c[min(a, b)] += 1; c[max(a, b)] += 1

  have_middle = any(N1+1 <= u <= N1+N2 and N1+1 <= v <= N1+N2 for u, v in edges)

  distance_L = bfs(edges, 0, N)
  distance_R = bfs(edges, N-1, N)

  D = distance_L[N-1]
  
  # Case 1: middle component was disconnected
  if not have_middle:
    return D + 1

  # Case 2: at least one component at left can be connected to right
  for u in range(0, N1):
    if c[u] < N1-1 and distance_L[u] == distance_L[N-1]: return D + 1

  # Case 3: left and right must each connect to different vertices in middle component
  u洵 = [ u for u in range(N1) if c[u] == N1-1 and distance_L[u] == distance_L[N-1] ]
  u_states = set(distance_L[u敜 for u in u洵)
  v洵 = [ v for v in range(N1, N) if c[v] == N2-1 and distance_R[v] == distance_R[0] ]
  v_states = set(distance_R[v敜 for v in v洵)


  if len(u_states) > 1 or len(v_states) > 1 or min(u_states) < max(v_states):
    return D + 2

  return D + 3

N1, N2, M = [int(s) for s in input().split()]
edges = [ [int(s)-1 for s in input().split()] for m in range(M) ]

solution = solve(N1, N2, M, edges);
print(solution)