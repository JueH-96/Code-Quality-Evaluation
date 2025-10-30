def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    h, w, D = map(int, data[0].split())
    grid = []
    for i in range(1, 1+h):
        grid.append(data[i].strip())
    
    floors = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                floors.append((i, j))
                
    n = len(floors)
    max_count = 0
    for i in range(n):
        for j in range(i+1, n):
            p = floors[i]
            q = floors[j]
            covered = set()
            for f in floors:
                d1 = abs(f[0] - p[0]) + abs(f[1] - p[1])
                d2 = abs(f[0] - q[0]) + abs(f[1] - q[1])
                if d1 <= D or d2 <= D:
                    covered.add(f)
            if len(covered) > max_count:
                max_count = len(covered)
                
    print(max_count)

if __name__ == "__main__":
    main()