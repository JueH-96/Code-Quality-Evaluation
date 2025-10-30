import sys
from collections import deque

def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        return
    N, K = map(int, data)
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    V = list(map(int, sys.stdin.readline().split()))
    mark = [0] * (N+1)
    for v in V:
        mark[v] = 1

    # We'll do a DFS in two-phase post-order to compute subtree counts
    subtree_count = [0] * (N+1)
    edges_count = 0
    stack = [(1, -1, 0)]  # (node, parent, state) state=0 -> pre, state=1 -> post
    while stack:
        node, parent, state = stack.pop()
        if state == 0:
            # pre-order: push post-order marker, then children
            stack.append((node, parent, 1))
            for c in adj[node]:
                if c == parent:
                    continue
                stack.append((c, node, 0))
        else:
            # post-order: children are done
            cnt = mark[node]
            for c in adj[node]:
                if c == parent:
                    continue
                cc = subtree_count[c]
                cnt += cc
                # if this child's subtree has some but not all marked nodes,
                # the edge node--c is in the minimal Steiner tree
                if 0 < cc < K:
                    edges_count += 1
            subtree_count[node] = cnt

    # Number of vertices in the minimal subtree = number_of_included_edges + 1
    print(edges_count + 1)

if __name__ == "__main__":
    main()