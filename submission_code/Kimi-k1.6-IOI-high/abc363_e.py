import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx +=1
    W = int(data[idx])
    idx +=1
    Y = int(data[idx])
    idx +=1
    
    A = []
    for _ in range(H):
        A_row = list(map(int, data[idx:idx+W]))
        A.append(A_row)
        idx += W
    
    INF = 1 << 60
    M = [[INF]*W for _ in range(H)]
    heap = []
    
    # Initialize border cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                M[i][j] = A[i][j]
                heapq.heappush(heap, (A[i][j], i, j))
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Dijkstra-like algorithm
    while heap:
        current_max, i, j = heapq.heappop(heap)
        if current_max > M[i][j]:
            continue
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                new_max = max(current_max, A[ni][nj])
                if new_max < M[ni][nj]:
                    M[ni][nj] = new_max
                    heapq.heappush(heap, (new_max, ni, nj))
    
    total = H * W
    max_Y = Y
    cnt = [0] * (max_Y + 2)  # 0 to max_Y
    
    for i in range(H):
        for j in range(W):
            m = M[i][j]
            if m <= max_Y:
                cnt[m] +=1
    
    prefix = [0]*(max_Y +1)
    for i in range(1, max_Y +1):
        prefix[i] = prefix[i-1] + cnt[i]
    
    for i in range(1, Y+1):
        print(total - prefix[i])

if __name__ == "__main__":
    main()