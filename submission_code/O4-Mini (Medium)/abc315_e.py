import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N = int(input())
    # prereqs[i] = list of books you must read before i
    prereqs = [[] for _ in range(N+1)]
    # dependents[u] = list of books that depend directly on u
    dependents = [[] for _ in range(N+1)]
    
    for i in range(1, N+1):
        data = list(map(int, input().split()))
        Ci = data[0]
        for p in data[1:]:
            prereqs[i].append(p)
            dependents[p].append(i)
    
    # Find all books needed (the dependency-closure of book 1)
    vis = [False] * (N+1)
    stack = [1]
    vis[1] = True
    while stack:
        u = stack.pop()
        for p in prereqs[u]:
            if not vis[p]:
                vis[p] = True
                stack.append(p)
    
    # Compute in-degrees only among the needed books (excluding book 1)
    in_deg = [0] * (N+1)
    for i in range(2, N+1):
        if vis[i]:
            # every prereq p of i must also be in vis[]
            in_deg[i] = len(prereqs[i])
    
    # Kahn's algorithm for topological sort on the needed subgraph
    q = deque()
    for i in range(2, N+1):
        if vis[i] and in_deg[i] == 0:
            q.append(i)
    
    result = []
    while q:
        u = q.popleft()
        result.append(u)
        for v in dependents[u]:
            if v != 1 and vis[v]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
    
    # Print the sequence (excluding book 1)
    print(" ".join(map(str, result)))

# Call main to execute
main()