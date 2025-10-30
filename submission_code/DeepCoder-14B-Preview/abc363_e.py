import heapq
import bisect

def main():
    H, W, Y = map(int, input().split())
    A = []
    for _ in range(H):
        row = list(map(int, input().split()))
        A.append(row)
    
    INF = 10**18
    s = [[INF for _ in range(W)] for __ in range(H)]
    heap = []
    
    # Initialize the perimeter cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                s[i][j] = A[i][j]
                heapq.heappush(heap, (s[i][j], i, j))
    
    # Directions: up, down, left, right
    dirs = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    while heap:
        current_s, i, j = heapq.heappop(heap)
        if current_s > s[i][j]:
            continue
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_s = max(current_s, A[ni][nj])
                if new_s < s[ni][nj]:
                    s[ni][nj] = new_s
                    heapq.heappush(heap, (new_s, ni, nj))
    
    # Collect all s values that are <= Y
    s_values = []
    for i in range(H):
        for j in range(W):
            if s[i][j] <= Y:
                s_values.append(s[i][j])
    
    # Sort the s_values
    s_values.sort()
    
    # For each year from 1 to Y, compute the result
    for i in range(1, Y+1):
        cnt = bisect.bisect_right(s_values, i)
        print(H * W - cnt)

if __name__ == '__main__':
    main()