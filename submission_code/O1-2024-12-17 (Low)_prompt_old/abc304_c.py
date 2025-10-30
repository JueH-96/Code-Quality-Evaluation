def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = int(input_data[1])
    coords = [(int(input_data[2 + 2*i]), int(input_data[3 + 2*i])) for i in range(N)]
    
    # Create adjacency list based on distance threshold
    adjacency = [[] for _ in range(N)]
    D_squared = D*D
    
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1, N):
            x_j, y_j = coords[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx*dx + dy*dy
            if dist_sq <= D_squared:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # We use BFS starting from person 1 (index 0)
    infected = [False]*N
    infected[0] = True
    
    from collections import deque
    queue = deque([0])
    
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if not infected[nxt]:
                infected[nxt] = True
                queue.append(nxt)
    
    # Output results
    for status in infected:
        print("Yes" if status else "No")

# Let's call solve() to handle the problem logic.
def main():
    solve()

if __name__ == "__main__":
    main()