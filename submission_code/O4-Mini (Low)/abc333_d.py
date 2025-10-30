import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # If 1 is already a leaf, just one operation.
    d = len(adj[1])
    if d <= 1:
        print(1)
        return
    
    # Mark 1 as visited; for each neighbor, BFS/DFS to count its component size.
    visited = [False] * (N+1)
    visited[1] = True
    sizes = []
    from collections import deque
    for nei in adj[1]:
        if not visited[nei]:
            cnt = 0
            dq = deque([nei])
            visited[nei] = True
            while dq:
                u = dq.popleft()
                cnt += 1
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        dq.append(w)
            sizes.append(cnt)
    # We have exactly d subtrees; to make 1 degree<=1 we delete (d-1) smallest subtrees.
    sizes.sort()
    ops = sum(sizes[:d-1]) + 1  # delete those nodes + finally delete node 1
    print(ops)

if __name__ == "__main__":
    main()