import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    Y = int(data[idx]); idx +=1
    
    A = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        A.append(row)
        idx += W
    
    INF = float('inf')
    m = [[INF]*W for _ in range(H)]
    heap = []
    
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                m[i][j] = A[i][j]
                heapq.heappush(heap, (m[i][j], i, j))
    
    dirs = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    while heap:
        current_m, i, j = heapq.heappop(heap)
        if current_m > m[i][j]:
            continue
        for di, dj in dirs:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbor_A = A[ni][nj]
                candidate_m = max(current_m, neighbor_A)
                if candidate_m < m[ni][nj]:
                    m[ni][nj] = candidate_m
                    heapq.heappush(heap, (m[ni][nj], ni, nj))
    
    max_m_val = 10**5
    freq = [0] * (max_m_val + 2)
    for i in range(H):
        for j in range(W):
            mv = m[i][j]
            if mv <= max_m_val:
                freq[mv] += 1
    
    prefix_sum = [0] * (max_m_val + 2)
    prefix_sum[0] = freq[0]
    for t in range(1, max_m_val +1):
        prefix_sum[t] = prefix_sum[t-1] + freq[t]
    
    total = H * W
    max_m = max_m_val
    for t in range(1, Y+1):
        if t <= max_m:
            ans = total - prefix_sum[t]
        else:
            ans = 0
        print(ans)
    
if __name__ == "__main__":
    main()