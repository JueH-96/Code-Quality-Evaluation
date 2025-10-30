from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N-1):
        u = int(data[index])
        v = int(data[index+1])
        edges.append((u, v))
        index += 2
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all leaves
    leaves = []
    for i in range(1, N+1):
        if len(adj[i]) == 1:
            leaves.append(i)
    
    # BFS to find the order of leaves
    order = []
    visited = [False] * (N+1)
    q = deque()
    q.append(leaves[0])
    visited[leaves[0]] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    # Now, find the pairs
    paired = set()
    levels = []
    for i in range(0, len(leaves), 2):
        u = leaves[i]
        v = leaves[i+1]
        paired.add(u)
        paired.add(v)
        levels.append(2)
    
    # Now, find the remaining leaves and their connections
    # Since the tree is formed by connecting leaves of stars, the remaining leaves are connected to the centers
    # So, the remaining leaves are connected to the same center, forming a larger star
    # So, the number of such leaves is the level of the star minus 2
    # Wait, perhaps it's easier to count the number of leaves and see how they are connected
    # Alternatively, we can find the centers of the stars
    # The centers are the nodes with degree greater than 2
    centers = []
    for i in range(1, N+1):
        if len(adj[i]) > 2:
            centers.append(i)
    
    # Now, for each center, count the number of leaves connected to it
    # The level of the star is the number of leaves connected to it plus 1
    for center in centers:
        count = 0
        for neighbor in adj[center]:
            if len(adj[neighbor]) == 1:
                count += 1
        levels.append(count + 1)
    
    # Sort the levels
    levels.sort()
    print(' '.join(map(str, levels)))

if __name__ == "__main__":
    main()