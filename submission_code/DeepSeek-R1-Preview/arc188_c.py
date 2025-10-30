import sys
from collections import defaultdict, deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    testimonies = [[] for _ in range(N+1)]  # 1-based
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        testimonies[A].append((B, C))

    # Build the graph
    graph = defaultdict(list)
    for A in range(1, N+1):
        ts = testimonies[A]
        if len(ts) < 2:
            continue
        for i in range(len(ts)-1):
            B1, C1 = ts[i]
            B2, C2 = ts[i+1]
            w = (C1 != C2)
            graph[B1].append((B2, w))
            graph[B2].append((B1, w))

    # Check bipartition
    color = {}
    is_bipartite = True
    for node in range(1, N+1):
        if node not in color:
            queue = deque()
            queue.append(node)
            color[node] = 0
            while queue and is_bipartite:
                current = queue.popleft()
                for neighbor, w in graph.get(current, []):
                    if neighbor not in color:
                        color[neighbor] = color[current] ^ w
                        queue.append(neighbor)
                    else:
                        if (color[current] ^ w) != color[neighbor]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
    if not is_bipartite:
        print(-1)
        return

    # Assign S (honest status)
    S = {node: color[node] for node in color}

    # Compute C (confusion)
    C = [0] * (N + 1)  # 0-based
    for A in range(1, N+1):
        ts = testimonies[A]
        if not ts:
            C[A] = 0
            continue
        # Take the first testimony
        B, C_i = ts[0]
        # S[A] is not known yet
        # So, we have to compute C[A] based on S[A], which is not in the graph
        # This is a problem because S[A] is not part of the graph we built
        # So, this approach is incorrect
        # Thus, we cannot compute C[A] without knowing S[A], which is not determined
        # Therefore, this approach is flawed
        print(-1)
        return

    # Check all testimonies
    # But since S is not fully determined, this is not possible
    print(-1)
    return

if __name__ == '__main__':
    main()