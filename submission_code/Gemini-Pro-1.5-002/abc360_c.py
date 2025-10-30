# YOUR CODE HERE
import sys

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    w = list(map(int, sys.stdin.readline().split()))

    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[a[i]-1].append(i)

    visited = [False] * n
    ans = 0

    for i in range(n):
        if visited[i]:
            continue

        cycle = []
        curr = i
        while not visited[curr]:
            visited[curr] = True
            cycle.append(curr)
            found = False
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    curr = neighbor
                    found = True
                    break
            if not found:
                break
        
        if len(cycle) > 1 and curr == i:
            min_weight = float('inf')
            total_weight = 0
            for node in cycle:
                min_weight = min(min_weight, w[node])
                total_weight += w[node]
            
            ans += total_weight - min_weight + min(min_weight, min([w[node] for node in cycle if node not in adj[node]] or [float('inf')]))

    print(ans)


solve()