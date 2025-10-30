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
        A.append(list(map(int, data[idx:idx+W])))
        idx += W
    
    INF = float('inf')
    M = [[INF]*W for _ in range(H)]
    heap = []
    
    # Push all border cells into the heap
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j ==0 or j == W-1:
                M[i][j] = A[i][j]
                heapq.heappush(heap, (A[i][j], i, j))
    
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    while heap:
        current_m, i, j = heapq.heappop(heap)
        if current_m > M[i][j]:
            continue
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_m = max(current_m, A[ni][nj])
                if new_m < M[ni][nj]:
                    M[ni][nj] = new_m
                    heapq.heappush(heap, (new_m, ni, nj))
    
    max_m_needed = Y
    freq = [0] * (max_m_needed + 2)  # 0 to Y+1
    
    for i in range(H):
        for j in range(W):
            m = M[i][j]
            if m <= max_m_needed:
                freq[m] +=1
    
    prefix = [0] * (max_m_needed + 2)
    for y in range(1, max_m_needed +1):
        prefix[y] = prefix[y-1] + freq[y]
    
    total = H * W
    for y in range(1, Y+1):
        print(total - prefix[y])

if __name__ == "__main__":
    main()