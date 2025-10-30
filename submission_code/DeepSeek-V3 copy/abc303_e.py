from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [[] for _ in range(N+1)]
    index = 1
    for _ in range(N-1):
        u = int(data[index])
        v = int(data[index+1])
        edges[u].append(v)
        edges[v].append(u)
        index += 2
    
    # Find all leaves
    leaves = []
    for i in range(1, N+1):
        if len(edges[i]) == 1:
            leaves.append(i)
    
    # BFS to find the order of leaves
    visited = [False] * (N+1)
    order = []
    q = deque()
    for leaf in leaves:
        q.append(leaf)
        visited[leaf] = True
    
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in edges[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)
    
    # Now, we need to find the pairs of leaves that were connected
    # Since the order is from leaves to the center, we can pair them in order
    paired = [False] * (N+1)
    L = []
    for i in range(0, len(leaves), 2):
        if i+1 < len(leaves):
            L.append(2)
    
    # Now, we need to handle the case where the number of leaves is odd
    # This happens when there is a star with level greater than 2
    # So, we need to find the center node and count the number of leaves connected to it
    # The center node is the one with degree equal to the number of leaves minus the number of pairs
    # Wait, perhaps a better approach is to find the node with the highest degree
    # The center node will have degree equal to the number of leaves minus the number of pairs
    
    # Let's find the center node
    center = -1
    max_deg = 0
    for i in range(1, N+1):
        if len(edges[i]) > max_deg:
            max_deg = len(edges[i])
            center = i
    
    # Now, count the number of leaves connected to the center
    count = 0
    for neighbor in edges[center]:
        if len(edges[neighbor]) == 1:
            count += 1
    
    # The level of the center star is count
    if count > 0:
        L.append(count)
    
    # Sort L in ascending order
    L.sort()
    print(' '.join(map(str, L)))

if __name__ == "__main__":
    main()