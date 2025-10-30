import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr += 1
    W = int(input[ptr]); ptr += 1
    Y = int(input[ptr]); ptr += 1

    A = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        A.append(row)

    INF = float('inf')
    d = [[INF] * W for _ in range(H)]
    heap = []

    # Initialize border cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                d[i][j] = A[i][j]
                heapq.heappush(heap, (d[i][j], i, j))

    # Dijkstra's algorithm to compute minimal maximum elevation
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    while heap:
        current_val, i, j = heapq.heappop(heap)
        if current_val > d[i][j]:
            continue
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_val = max(current_val, A[ni][nj])
                if new_val < d[ni][nj]:
                    d[ni][nj] = new_val
                    heapq.heappush(heap, (new_val, ni, nj))

    # Calculate frequency array
    freq = [0] * (Y + 2)  # freq[0...Y+1]
    for i in range(H):
        for j in range(W):
            val = d[i][j]
            if val <= Y:
                freq[val] += 1

    # Compute prefix sum
    prefix = [0] * (Y + 2)
    for i in range(1, Y + 1):
        prefix[i] = prefix[i-1] + freq[i]

    # Output results
    total = H * W
    for i in range(1, Y + 1):
        print(total - prefix[i])

if __name__ == "__main__":
    main()