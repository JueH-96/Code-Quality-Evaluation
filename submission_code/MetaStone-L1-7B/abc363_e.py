import heapq

def main():
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    
    sink_time = [[float('inf')] * W for _ in range(H)]
    heap = []
    
    # Initialize edges
    for x in range(H):
        for y in [0, W-1]:
            sink_time[x][y] = A[x][y]
            heapq.heappush(heap, (A[x][y], x, y))
    for y in range(W):
        for x in [0, H-1]:
            sink_time[x][y] = A[x][y]
            heapq.heappush(heap, (A[x][y], x, y))
    
    # Directions: up, down, left, right
    dirs = [ (-1, 0), (1, 0), (0, -1), (0, 1) ]
    
    while heap:
        curr_max, x, y = heapq.heappop(heap)
        if curr_max > sink_time[x][y]:
            continue
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                new_max = max(curr_max, A[nx][ny])
                if new_max < sink_time[nx][ny]:
                    sink_time[nx][ny] = new_max
                    heapq.heappush(heap, (new_max, nx, ny))
    
    # Compute frequency and prefix sum
    max_i = Y
    freq = [0] * (max_i + 2)
    
    for x in range(H):
        for y in range(W):
            m = sink_time[x][y]
            if m <= max_i:
                freq[m] += 1
    
    # Compute prefix sums
    sum_ = [0] * (max_i + 2)
    for i in range(1, max_i + 1):
        sum_[i] = sum_[i-1] + freq[i]
    
    # Output the results
    for i in range(1, Y + 1):
        print(H * W - sum_[i])

if __name__ == "__main__":
    main()