import heapq

def main():
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    INF = float('inf')
    
    # Initialize D matrix with infinity
    D = [[INF for _ in range(W)] for __ in range(H)]
    heap = []
    
    # Set border cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                D[i][j] = A[i][j]
                heapq.heappush(heap, (D[i][j], i, j))
    
    # Directions for neighbors
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Process the priority queue
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > D[i][j]:
            continue
        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                new_d = max(d, A[ni][nj])
                if new_d < D[ni][nj]:
                    D[ni][nj] = new_d
                    heapq.heappush(heap, (new_d, ni, nj))
    
    # Compute earliest_t and collect frequencies
    freq = {}
    max_t = 0
    for i in range(H):
        for j in range(W):
            t = max(D[i][j], A[i][j])
            if t > max_t:
                max_t = t
            if t in freq:
                freq[t] += 1
            else:
                freq[t] = 1
    
    # Compute suffix sums
    s = [0] * (max_t + 2)
    for i in range(max_t - 1, 0, -1):
        s[i] = s[i + 1] + freq.get(i + 1, 0)
    
    # Generate output for each year
    for year in range(1, Y + 1):
        if year > max_t:
            print(0)
        else:
            print(s[year])

if __name__ == "__main__":
    main()