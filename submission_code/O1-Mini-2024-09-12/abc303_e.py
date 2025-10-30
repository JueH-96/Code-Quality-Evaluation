# YOUR CODE HERE
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N,*rest = map(int, sys.stdin.read().split())
    edges = rest
    adj = [[] for _ in range(N+1)]
    degrees = [0]*(N+1)
    for i in range(0, len(edges),2):
        u, v = edges[i], edges[i+1]
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] +=1
        degrees[v] +=1
    centers = []
    for node in range(1, N+1):
        if degrees[node] >=2:
            has_leaf = False
            for neighbor in adj[node]:
                if degrees[neighbor]==1:
                    has_leaf=True
                    break
            if has_leaf:
                centers.append(degrees[node])
    centers.sort()
    print(' '.join(map(str, centers)))

if __name__ == "__main__":
    main()