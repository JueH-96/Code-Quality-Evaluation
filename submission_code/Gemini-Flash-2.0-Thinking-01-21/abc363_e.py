import collections

def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))
    
    results = []
    for year in range(1, y + 1):
        sea_level = year
        is_sunk = [[False for _ in range(w)] for _ in range(h)]
        q = collections.deque()
        
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    if a[r][c] <= sea_level:
                        is_sunk[r][c] = True
                        q.append((r, c))
                        
        while q:
            row, col = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w and not is_sunk[nr][nc] and a[nr][nc] <= sea_level:
                    is_sunk[nr][nc] = True
                    q.append((nr, nc))
                    
        sunk_count = 0
        for r in range(h):
            for c in range(w):
                if is_sunk[r][c]:
                    sunk_count += 1
                    
        remaining_area = h * w - sunk_count
        results.append(remaining_area)
        
    for area in results:
        print(area)

if __name__ == '__main__':
    solve()