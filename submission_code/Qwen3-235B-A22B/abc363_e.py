import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1

    A = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        A.append(row)
        idx += W

    INF = float('inf')
    critical = [[INF] * W for _ in range(H)]
    heap = []

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize boundary cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                c = A[i][j]
                critical[i][j] = c
                heapq.heappush(heap, (c, i, j))

    # Dijkstra-like algorithm to compute critical values
    while heap:
        current_c, x, y = heapq.heappop(heap)
        if current_c > critical[x][y]:
            continue
        for dx, dy in dirs:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < H and 0 <= ny < W:
                candidate = max(A[nx][ny], current_c)
                if candidate < critical[nx][ny]:
                    critical[nx][ny] = candidate
                    heapq.heappush(heap, (candidate, nx, ny))

    # Build frequency array
    max_val = 10**5 + 2
    freq = [0] * (max_val + 2)
    for i in range(H):
        for j in range(W):
            c = critical[i][j]
            if c <= max_val:
                freq[c] += 1

    # Build prefix sum array
    pre_sum = [0] * (max_val + 2)
    for s in range(1, max_val + 1):
        pre_sum[s] = pre_sum[s-1] + freq[s]

    # Output results for each year up to Y
    for s in range(1, Y+1):
        submerged = pre_sum[s]
        print(H * W - submerged)

if __name__ == "__main__":
    main()