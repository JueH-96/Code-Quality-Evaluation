import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    K = int(input[ptr])
    ptr +=1

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(input[ptr])
        ptr +=1
        b = int(input[ptr])
        ptr +=1
        adj[a].append(b)
        adj[b].append(a)

    # Read required nodes
    required = list(map(int, input[ptr:ptr+K]))
    ptr += K
    required_set = set(required)

    # Compute subtree counts using iterative post-order traversal
    root = required[0] if K > 0 else 1  # Choose the first required node as root to ensure required nodes are in the subtree
    subtree_count = [0] * (N + 1)
    stack = [(root, -1, False)]

    while stack:
        u, p, visited = stack.pop()
        if not visited:
            stack.append((u, p, True))
            # Push children in reverse order to process them in order
            for v in reversed(adj[u]):
                if v != p:
                    stack.append((v, u, False))
        else:
            cnt = 0
            if u in required_set:
                cnt += 1
            for v in adj[u]:
                if v != p:
                    cnt += subtree_count[v]
            subtree_count[u] = cnt

    # Calculate edge_count
    edge_count = 0
    for u in range(1, N + 1):
        if u == root:
            continue
        s = subtree_count[u]
        if 1 <= s <= K - 1:
            edge_count += 1

    print(edge_count + 1)

if __name__ == "__main__":
    main()