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
        row = list(map(int, data[idx:idx+W]))
        idx += W
        A.append(row)
    
    INF = float('inf')
    s_min = [[INF for _ in range(W)] for __ in range(H)]
    heap = []
    
    # Initialize border cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                s_min[i][j] = A[i][j]
                heapq.heappush(heap, (s_min[i][j], i, j))
    
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while heap:
        current_s, i, j = heapq.heappop(heap)
        if current_s > s_min[i][j]:
            continue
        for di, dj in dirs:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                candidate_s = max(current_s, A[ni][nj])
                if candidate_s < s_min[ni][nj]:
                    s_min[ni][nj] = candidate_s
                    heapq.heappush(heap, (candidate_s, ni, nj))
    
    # Collect all s_min values
    all_s = []
    for row in s_min:
        for s in row:
            all_s.append(s)
    
    # Prepare frequency array
    max_s = max(all_s)
    freq_size = min(Y, int(max_s)) + 1 if max_s != INF else Y + 1
    freq = [0] * (Y + 2)  # indexes 0 to Y+1
    
    for s in all_s:
        if s <= Y:
            freq[s] += 1
    
    # Compute prefix sums
    sum_freq = [0] * (Y + 2)
    current_sum = 0
    for i in range(1, Y + 1):
        current_sum += freq[i]
        sum_freq[i] = current_sum
    
    # Calculate and print results
    total_area = H * W
    for i in range(1, Y + 1):
        print(total_area - sum_freq[i])
    
if __name__ == '__main__':
    main()