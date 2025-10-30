import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    W = int(data[idx+1])
    Y = int(data[idx+2])
    idx +=3
    A = []
    for i in range(H):
        row = list(map(int, data[idx:idx+W]))
        A.append(row)
        idx += W
    
    # Initialize k matrix and priority queue
    k = [[0]*W for _ in range(H)]
    heap = []
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                heapq.heappush(heap, (A[i][j], i, j))
                k[i][j] = A[i][j]
    
    # Directions: up, down, left, right
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while heap:
        current_k, i, j = heapq.heappop(heap)
        # Skip if this entry is not the minimal k for (i,j)
        if current_k > k[i][j]:
            continue
        for dx, dy in dirs:
            x = i + dx
            y = j + dy
            if 0 <= x < H and 0 <= y < W:
                new_k = max(A[x][y], current_k)
                if new_k < k[x][y]:
                    k[x][y] = new_k
                    heapq.heappush(heap, (new_k, x, y))
    
    # Find maximum k value
    max_k = max(max(row) for row in k)
    
    # Create cnt array
    cnt = [0] * (max_k + 2)
    for i in range(H):
        for j in range(W):
            cnt[k[i][j]] += 1
    
    # Compute suffix sums
    sum_cnt = [0] * (max_k + 2)
    sum_cnt[max_k] = 0
    for y in range(max_k -1, -1, -1):
        sum_cnt[y] = sum_cnt[y+1] + cnt[y+1]
    
    # Generate output
    for year in range(1, Y+1):
        if year >= max_k:
            print(0)
        else:
            print(sum_cnt[year])

if __name__ == "__main__":
    main()