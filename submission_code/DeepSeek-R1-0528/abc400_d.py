import collections
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(list(data[i].strip()))
    
    parts = data[1+H].split()
    A = int(parts[0]) - 1
    B = int(parts[1]) - 1
    C = int(parts[2]) - 1
    D = int(parts[3]) - 1

    INF = 10**9
    dist = [[INF] * W for _ in range(H)]
    dist[A][B] = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    q = collections.deque()
    q.append((0, A, B))
    
    while q:
        kicks, r, c = q.popleft()
        if r == C and c == D:
            print(kicks)
            return
            
        if kicks > dist[r][c]:
            continue
            
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.':
                if dist[nr][nc] > kicks:
                    dist[nr][nc] = kicks
                    q.appendleft((kicks, nr, nc))
                    
        for dr, dc in directions:
            for step in (1, 2):
                nr = r + dr * step
                nc = c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'
            nr1 = r + dr
            nc1 = c + dc
            if 0 <= nr1 < H and 0 <= nc1 < W and grid[nr1][nc1] == '.':
                if dist[nr1][nc1] > kicks + 1:
                    dist[nr1][nc1] = kicks + 1
                    q.append((kicks + 1, nr1, nc1))
                    
    print(dist[C][D] if dist[C][D] != INF else -1)

if __name__ == "__main__":
    main()