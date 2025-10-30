def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    H, W = map(int, input_data[:2])
    grid = input_data[2:]
    
    # The pattern we're looking for
    pattern = "snuke"
    
    # dp[i][j][p] will be True if we can reach cell (i,j) (0-indexed internally)
    # with the letter index p (meaning grid[i][j] == pattern[p]).
    dp = [[[False]*5 for _ in range(W)] for __ in range(H)]
    
    # Initialize if the top-left cell matches 's' (pattern[0])
    if grid[0][0] == pattern[0]:
        dp[0][0][0] = True
    
    # BFS queue will hold tuples (row, col, p)
    queue = deque()
    if dp[0][0][0]:
        queue.append((0, 0, 0))
    
    # Directions for moving up/down/left/right
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while queue:
        i, j, p = queue.popleft()
        next_p = (p + 1) % 5
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                # We want the next cell to have the character pattern[next_p]
                if not dp[ni][nj][next_p] and grid[ni][nj] == pattern[next_p]:
                    dp[ni][nj][next_p] = True
                    queue.append((ni, nj, next_p))
    
    # Check if any index p is set to True at (H-1, W-1) and matches the letter
    # but that is automatically enforced by dp's definition.
    # We just need dp[H-1][W-1][p] to be True for the correct p that meets the pattern.
    # Actually, if dp[H-1][W-1][p] is True, that means grid[H-1][W-1] == pattern[p].
    # Hence at least one p must be True there.
    answer = any(dp[H-1][W-1][p] for p in range(5))
    
    print("Yes" if answer else "No")

# Don't forget to call main!
if __name__ == "__main__":
    main()