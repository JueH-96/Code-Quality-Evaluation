from collections import deque
import sys
sys.setrecursionlimit(10**7)

def main():
    input_data = sys.stdin.read().strip().split()
    N1, N2, M = map(int, input_data[:3])
    edges = input_data[3:]

    # Adjacency lists for the subgraphs:
    # subgraph1: vertices 1..N1 (1-based indexing)
    # subgraph2: vertices N1+1..N1+N2, we will remap them to 1..N2
    adj1 = [[] for _ in range(N1+1)]
    adj2 = [[] for _ in range(N2+1)]

    idx = 0
    for _ in range(M):
        a = int(edges[idx]); b = int(edges[idx+1])
        idx += 2
        if a <= N1 and b <= N1:
            # belongs to subgraph1
            adj1[a].append(b)
            adj1[b].append(a)
        elif a > N1 and b > N1:
            # belongs to subgraph2 (remap)
            a2 = a - N1
            b2 = b - N1
            adj2[a2].append(b2)
            adj2[b2].append(a2)
        else:
            # cross-edge would connect 1..N1 and N1+1..N1+N2
            # but the problem states 1 and N1+N2 remain disconnected,
            # so no valid path across. We ignore it (it cannot exist
            # in a well-formed input per problem constraints).
            pass

    # BFS in subgraph1 from vertex 1
    dist1 = [-1]*(N1+1)
    dist1[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for nxt in adj1[u]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[u] + 1
                queue.append(nxt)
    max_dist1 = max(dist1[1:])  # ignore dist1[0]

    # BFS in subgraph2 from vertex N2 (which corresponds to original vertex N1+N2)
    dist2 = [-1]*(N2+1)
    dist2[N2] = 0
    queue = deque([N2])
    while queue:
        u = queue.popleft()
        for nxt in adj2[u]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[u] + 1
                queue.append(nxt)
    max_dist2 = max(dist2[1:])  # ignore dist2[0]

    # The answer is max_dist1 + 1 + max_dist2
    print(max_dist1 + max_dist2 + 1)

# Do not forget to call main()
if __name__ == "__main__":
    main()