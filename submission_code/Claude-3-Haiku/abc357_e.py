from collections import defaultdict

def count_reachable_pairs(n, a):
    # Create a graph representation
    graph = defaultdict(list)
    for i in range(n):
        graph[i+1].append(a[i])

    # Perform DFS to find reachable vertices from each vertex
    reachable = [set() for _ in range(n+1)]
    stack = []
    for u in range(1, n+1):
        if u not in reachable[u]:
            stack.append(u)
            reachable[u].add(u)
            while stack:
                v = stack.pop()
                for w in graph[v]:
                    if w not in reachable[u]:
                        reachable[u].add(w)
                        stack.append(w)

    # Count the number of reachable pairs
    count = 0
    for u in range(1, n+1):
        count += len(reachable[u])
    return count

# Read input
n = int(input())
a = [int(x) for x in input().split()]

# Solve the problem
print(count_reachable_pairs(n, a))