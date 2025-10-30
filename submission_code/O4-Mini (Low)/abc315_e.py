import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    # prereq[i]: list of books that must be read before i
    prereq = [[] for _ in range(N+1)]
    # graph: edges from prereq -> dependent
    graph = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        data = input().split()
        c = int(data[0])
        if c:
            ps = list(map(int, data[1:]))
            prereq[i] = ps
            for p in ps:
                graph[p].append(i)

    # Find all ancestors of book 1 (books needed to read before 1)
    needed = [False] * (N+1)
    stack = [1]
    needed[1] = True
    while stack:
        u = stack.pop()
        for p in prereq[u]:
            if not needed[p]:
                needed[p] = True
                stack.append(p)

    # Compute in-degrees within the needed subgraph
    indeg = [0] * (N+1)
    for v in range(1, N+1):
        if not needed[v]: 
            continue
        # count only prereqs that are also needed
        for p in prereq[v]:
            if needed[p]:
                indeg[v] += 1

    # Kahn's algorithm: start with needed nodes (excluding 1) of indeg 0
    from collections import deque
    q = deque()
    for v in range(1, N+1):
        if needed[v] and v != 1 and indeg[v] == 0:
            q.append(v)

    order = []
    while q:
        u = q.popleft()
        order.append(u)
        # for each dependent of u
        for v in graph[u]:
            if not needed[v]:
                continue
            indeg[v] -= 1
            if v != 1 and indeg[v] == 0:
                q.append(v)

    # Print the reading order (excluding book 1)
    print(" ".join(map(str, order)))

if __name__ == "__main__":
    main()