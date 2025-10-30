def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = int(data[1])
    coords = [(int(data[2+2*i]), int(data[3+2*i])) for i in range(N)]
    
    # We'll use squared distances to compare with D^2 to avoid unnecessary floating ops
    D2 = D * D
    
    # Build adjacency list
    # adjacency[i] will store the list of indices j where distance(i, j) <= D
    adjacency = [[] for _ in range(N)]
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1, N):
            x_j, y_j = coords[j]
            dx, dy = x_i - x_j, y_i - y_j
            dist2 = dx*dx + dy*dy
            if dist2 <= D2:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Perform BFS or DFS from person 1 (index 0)
    infected = [False]*N
    infected[0] = True
    
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if not infected[neighbor]:
                infected[neighbor] = True
                queue.append(neighbor)
    
    # Output results
    for i in range(N):
        print("Yes" if infected[i] else "No")

# Do not forget to call main()
main()