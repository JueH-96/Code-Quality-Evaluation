def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    N, D = int(data[0]), int(data[1])
    coords = []
    
    idx = 2
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        coords.append((x, y))
        idx += 2

    # Precompute adjacency based on distance <= D
    adj_list = [[] for _ in range(N)]
    D_squared = D * D

    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1, N):
            x_j, y_j = coords[j]
            dx = x_i - x_j
            dy = y_i - y_j
            if dx*dx + dy*dy <= D_squared:
                adj_list[i].append(j)
                adj_list[j].append(i)

    # BFS from person 1 (index 0 in 0-based) to find infected persons
    infected = [False] * N
    infected[0] = True
    queue = deque([0])

    while queue:
        current = queue.popleft()
        for neighbor in adj_list[current]:
            if not infected[neighbor]:
                infected[neighbor] = True
                queue.append(neighbor)

    # Output result
    for i in range(N):
        print("Yes" if infected[i] else "No")

# Do not forget to call main at the end
if __name__ == "__main__":
    main()