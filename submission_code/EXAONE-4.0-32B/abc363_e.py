import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    A = []
    index = 3
    for i in range(H):
        row = list(map(int, data[index:index+W]))
        A.append(row)
        index += W
        
    total = H * W
    BIG = 10**6
    d = [[BIG] * W for _ in range(H)]
    heap = []
    
    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                d[i][j] = A[i][j]
                heapq.heappush(heap, (d[i][j], i, j))
                
    while heap:
        val, i, j = heapq.heappop(heap)
        if val != d[i][j]:
            continue
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                new_val = max(val, A[ni][nj])
                if new_val < d[ni][nj]:
                    d[ni][nj] = new_val
                    heapq.heappush(heap, (new_val, ni, nj))
                    
    max_level = 100000
    freq = [0] * (max_level + 2)
    
    for i in range(H):
        for j in range(W):
            if d[i][j] <= max_level:
                freq[d[i][j]] += 1
                
    F = [0] * (max_level + 2)
    for level in range(1, max_level + 1):
        F[level] = F[level - 1] + freq[level]
        
    out_lines = []
    for year in range(1, Y + 1):
        sunk = F[year]
        remaining = total - sunk
        out_lines.append(str(remaining))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()