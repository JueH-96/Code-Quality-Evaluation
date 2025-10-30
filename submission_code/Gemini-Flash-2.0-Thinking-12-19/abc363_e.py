import collections

def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))
    
    is_sunk = [[False for _ in range(w)] for _ in range(h)]
    
    results = []
    for year in range(1, y + 1):
        sea_level = year
        will_sink_this_year = [[False for _ in range(w)] for _ in range(h)]
        queue = collections.deque()
        
        for i in range(h):
            for j in range(w):
                is_boundary = (i == 0 or i == h - 1 or j == 0 or j == w - 1)
                if is_boundary and a[i][j] <= sea_level and not is_sunk[i][j] and not will_sink_this_year[i][j]:
                    queue.append((i, j))
                    will_sink_this_year[i][j] = True
                    
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not is_sunk[nr][nc] and a[nr][nc] <= sea_level and not will_sink_this_year[nr][nc]:
                        queue.append((nr, nc))
                        will_sink_this_year[nr][nc] = True
                        
        sunk_count_this_year = 0
        for i in range(h):
            for j in range(w):
                if will_sink_this_year[i][j]:
                    if not is_sunk[i][j]:
                        is_sunk[i][j] = True
                        sunk_count_this_year += 1
                        
        remaining_area = 0
        for i in range(h):
            for j in range(w):
                if not is_sunk[i][j]:
                    remaining_area += 1
        results.append(remaining_area)
        
    for area in results:
        print(area)

if __name__ == '__main__':
    solve()