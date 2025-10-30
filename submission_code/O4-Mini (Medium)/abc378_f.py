import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for _ in range(N-1):
        u,v = map(int,input().split())
        adj[u].append(v)
        adj[v].append(u)
        deg[u]+=1
        deg[v]+=1

    visited = [False]*(N+1)
    ans = 0

    # We find connected components of deg==3 nodes,
    # For each component, count how many deg==2 neighbors it has.
    # Any pair among those deg2 neighbors yields a valid cycle.
    for i in range(1, N+1):
        if deg[i]==3 and not visited[i]:
            # DFS this deg3-component
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if deg[w]==3 and not visited[w]:
                        visited[w] = True
                        stack.append(w)
                        comp.append(w)
            # now comp holds all nodes of this deg3-component
            # collect all deg2 neighbors
            cnt2 = 0
            # to be safe, track seen deg2 neighbors
            seen2 = set()
            for u in comp:
                for w in adj[u]:
                    if deg[w]==2 and w not in seen2:
                        seen2.add(w)
            k = len(seen2)
            if k >= 2:
                ans += k*(k-1)//2

    print(ans)

if __name__ == "__main__":
    main()