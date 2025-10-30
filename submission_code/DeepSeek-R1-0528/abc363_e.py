import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it)); W = int(next(it)); Y_val = int(next(it))
    A = []
    for _ in range(H):
        row = [int(next(it)) for _ in range(W)]
        A.append(row)
    
    total = H * W
    max_t = 100000
    INF = 10**9 + 10
    dist = [[INF] * W for _ in range(H)]
    buckets = [[] for _ in range(max_t + 1)]
    
    for i in [0, H-1]:
        for j in range(W):
            d0 = A[i][j]
            if d0 < dist[i][j]:
                dist[i][j] = d0
                buckets[d0].append((i, j))
    
    for j in [0, W-1]:
        for i in range(1, H-1):
            d0 = A[i][j]
            if d0 < dist[i][j]:
                dist[i][j] = d0
                buckets[d0].append((i, j))
    
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for d_val in range(0, max_t + 1):
        if not buckets[d_val]:
            continue
        q = deque(buckets[d_val])
        while q:
            i, j = q.popleft()
            if dist[i][j] != d_val:
                continue
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    new_d = max(d_val, A[ni][nj])
                    if new_d < dist[ni][nj]:
                        dist[ni][nj] = new_d
                        if new_d == d_val:
                            q.append((ni, nj))
                        else:
                            buckets[new_d].append((ni, nj))
    
    freq = [0] * (max_t + 1)
    for i in range(H):
        for j in range(W):
            t = dist[i][j]
            if t <= max_t:
                freq[t] += 1
    
    cumulative = [0] * (max_t + 1)
    if max_t >= 1:
        cumulative[0] = 0
        for t in range(1, max_t + 1):
            cumulative[t] = cumulative[t-1] + freq[t]
    
    for year in range(1, Y_val + 1):
        if year > max_t:
            sunk_count = cumulative[max_t]
        else:
            sunk_count = cumulative[year]
        remaining = total - sunk_count
        print(remaining)

if __name__ == '__main__':
    main()