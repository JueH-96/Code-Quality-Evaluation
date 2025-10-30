import heapq
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(list(data[i].strip()))
    
    coords = list(map(int, data[1+H].split()))
    A, B, C, D = [x-1 for x in coords]
    
    INF = 10**9
    min_kicks = [[INF] * W for _ in range(H)]
    min_kicks[A][B] = 0
    heap = [(0, A, B)]
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while heap:
        k, r, c = heapq.heappop(heap)
        if r == C and c == D:
            print(k)
            return
        
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] == '.':
                    if k < min_kicks[nr][nc]:
                        min_kicks[nr][nc] = k
                        heapq.heappush(heap, (k, nr, nc))
        
        for dr, dc in dirs:
            broke = False
            for step in [1, 2]:
                nr = r + dr * step
                nc = c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'
                        broke = True
            if broke:
                heapq.heappush(heap, (k+1, r, c))
                
    print(-1)

if __name__ == "__main__":
    main()