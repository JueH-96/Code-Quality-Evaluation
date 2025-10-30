def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    # Read input all at once and parse
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    D = int(data[1])
    points = []
    index = 2
    for _ in range(N):
        x = int(data[index])
        y = int(data[index+1])
        points.append((x, y))
        index += 2

    # Use squared distance to avoid unnecessary square roots
    D2 = D * D

    # Setup union-find structure
    parent = list(range(N))
    
    def find(x):
        # Path compression optimization.
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # For each pair of persons, if within distance D, union them.
    for i in range(N):
        xi, yi = points[i]
        for j in range(i + 1, N):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= D2:
                union(i, j)
    
    # Person 1 (index 0) starts infected. 
    # If person i is in the same union as person 1, then person i is infected.
    rep_1 = find(0)
    output = []
    for i in range(N):
        if find(i) == rep_1:
            output.append("Yes")
        else:
            output.append("No")
    
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()