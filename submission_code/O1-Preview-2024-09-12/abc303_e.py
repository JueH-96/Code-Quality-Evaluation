# YOUR CODE HERE
import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    from collections import defaultdict, deque

    adj = defaultdict(list)
    degrees = [0] * (N + 1)
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] +=1
        degrees[v] +=1

    levels = []
    visited = [False] * (N + 1)

    def dfs(u):
        visited[u] = True
        size = 1
        for v in adj[u]:
            if not visited[v] and degrees[v] ==2:
                size += dfs(v)
        return size

    for u in range(1, N+1):
        if not visited[u]:
            if degrees[u] >=3:
                level = degrees[u]
                levels.append(level)
                visited[u] = True
                for v in adj[u]:
                    if degrees[v]==2 and not visited[v]:
                        dfs(v)
            elif degrees[u]==1:
                # check if it's an isolated leaf
                level = 2
                levels.append(level)
                visited[u] = True
                for v in adj[u]:
                    if degrees[v]==2 and not visited[v]:
                        dfs(v)
            elif degrees[u] ==2:
                # start DFS if not visited
                size = dfs(u)
                if size >=1 and not any(degrees[v]>=3 for v in adj[u]):
                    level = size +1
                    levels.append(level)
    levels.sort()
    print(' '.join(map(str, levels)))

threading.Thread(target=main).start()