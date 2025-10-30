import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); K = int(line[1])
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a,b = map(int, data.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    specials = list(map(int, data.readline().split()))
    is_special = [0]*(N+1)
    for v in specials:
        is_special[v] = 1

    # Build a DFS order and parent array (iterative)
    parent = [0]*(N+1)
    order = []
    stack = [1]
    parent[1] = 0
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            stack.append(v)

    # Process in reverse order to accumulate counts of specials in subtrees
    cnt = [0]*(N+1)
    total_edges = 0
    for u in reversed(order):
        # include this node's special flag
        cnt[u] += is_special[u]
        if u != 1:
            # if edge (parent[u], u) lies on a path between specials
            if 0 < cnt[u] < K:
                total_edges += 1
            # propagate count upward
            cnt[parent[u]] += cnt[u]

    # number of vertices = edges in the minimal subtree + 1
    answer = total_edges + 1
    print(answer)

if __name__ == "__main__":
    main()