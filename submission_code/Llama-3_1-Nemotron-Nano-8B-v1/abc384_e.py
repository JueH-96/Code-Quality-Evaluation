import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    P = int(input[idx])-1; idx +=1  # converting to 0-based
    Q = int(input[idx])-1; idx +=1
    
    grid = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        idx += W
        grid.append(row)
    
    # Initialize DP table and priority queue
    dp = [[0]*W for _ in range(H)]
    heap = []
    initial_strength = grid[P][Q]
    dp[P][Q] = initial_strength
    heapq.heappush(heap, (-initial_strength, P, Q))  # use negative for max-heap
    
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    max_strength = initial_strength
    
    while heap:
        current_strength_neg, i, j = heapq.heappop(heap)
        current_strength = -current_strength_neg
        
        if current_strength < dp[i][j]:
            continue  # Skip if a better strength is already recorded
        
        for di, dj in dirs:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                s = grid[ni][nj]
                if s < current_strength / X:
                    new_strength = current_strength + s
                    if new_strength > dp[ni][nj]:
                        dp[ni][nj] = new_strength
                        heapq.heappush(heap, (-new_strength, ni, nj))
                        if new_strength > max_strength:
                            max_strength = new_strength
    
    print(max_strength)

if __name__ == "__main__":
    main()