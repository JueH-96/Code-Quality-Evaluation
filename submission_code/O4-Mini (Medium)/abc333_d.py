import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    parent = [0]*(N+1)
    order = []
    stack = [1]
    parent[1] = -1  # mark root

    # iterative DFS to get an order and parent[]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            stack.append(v)

    # compute subtree sizes bottom-up
    size = [1]*(N+1)
    for u in reversed(order):
        for v in adj[u]:
            if v == parent[u]:
                continue
            size[u] += size[v]

    # among the neighbors of 1, find the largest subtree size
    max_sub = 0
    for v in adj[1]:
        if size[v] > max_sub:
            max_sub = size[v]

    # answer is N - max_sub
    print(N - max_sub)

if __name__ == "__main__":
    main()