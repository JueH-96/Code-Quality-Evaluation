# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    degrees = [0]*(N+1)
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] +=1
        degrees[v] +=1
    # BFS from vertex1
    distance = [0]*(N+1)
    visited = [False]*(N+1)
    q = deque()
    q.append(1)
    visited[1]=True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                distance[v] = distance[u]+1
                visited[v]=True
                q.append(v)
    # Find all leaves
    leaves = []
    for i in range(1,N+1):
        if degrees[i]==1:
            leaves.append(i)
    if degrees[1]==1 and N!=1:
        # vertex1 is a leaf
        print(1)
        return
    # Find maximum distance among leaves
    max_dist = -1
    count =0
    for leaf in leaves:
        if leaf ==1:
            continue
        if distance[leaf] > max_dist:
            max_dist = distance[leaf]
            count =1
        elif distance[leaf] == max_dist:
            count +=1
    if max_dist == -1:
        # only vertex1 exists
        print(1)
    else:
        print(max_dist + count)

if __name__ == "__main__":
    main()