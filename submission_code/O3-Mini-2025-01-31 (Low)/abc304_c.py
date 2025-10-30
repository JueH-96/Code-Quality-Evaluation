def main():
    import sys
    import math
    input = sys.stdin.readline

    # Read N (number of people) and D (infection distance threshold)
    N, D = map(int, input().split())
    D2 = D * D  # We'll compare squared distances to avoid computing square roots

    # Read each person's coordinates. Indexed 0 through N-1.
    coords = [tuple(map(int, input().split())) for _ in range(N)]

    # Build graph as an adjacency list. Two persons i and j are connected if the squared distance <= D2.
    # Since infection spreads by connectivity, we'll do a DFS/BFS from person 1 (index 0).
    # Instead of building the entire graph explicitly, we can use on-the-fly computation.
    # But given D and N, O(N^2) is acceptable.
    
    # Create list of bool infected for each person.
    infected = [False] * N
    infected[0] = True

    # Use a queue for BFS
    from collections import deque
    queue = deque([0])
    
    # BFS to mark infected persons
    while queue:
        i = queue.popleft()
        xi, yi = coords[i]
        # Check every person for connectivity from person i.
        # We can traverse all persons j that are not infected.
        for j in range(N):
            if not infected[j]:
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy <= D2:
                    infected[j] = True
                    queue.append(j)
                    
    # Output result
    out = []
    for inf in infected:
        out.append("Yes" if inf else "No")
    
    sys.stdout.write("
".join(out))
    
if __name__ == '__main__':
    main()