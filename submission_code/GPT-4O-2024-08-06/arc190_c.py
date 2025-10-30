# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def main():
    data = input().split()
    index = 0
    
    H = int(data[index])
    W = int(data[index + 1])
    index += 2
    
    A = []
    for _ in range(H):
        A.append([int(data[index + i]) for i in range(W)])
        index += W
    
    Q = int(data[index])
    sh = int(data[index + 1]) - 1
    sw = int(data[index + 2]) - 1
    index += 3
    
    queries = []
    for _ in range(Q):
        d = data[index]
        a = int(data[index + 1])
        queries.append((d, a))
        index += 2
    
    # Initialize dp array
    dp = [[0] * W for _ in range(H)]
    dp[0][0] = A[0][0]
    
    # Fill the dp array
    for h in range(H):
        for w in range(W):
            if h == 0 and w == 0:
                continue
            if h > 0:
                dp[h][w] += dp[h-1][w]
            if w > 0:
                dp[h][w] += dp[h][w-1]
            dp[h][w] = (dp[h][w] * A[h][w]) % MOD
    
    # Process each query
    current_h = sh
    current_w = sw
    
    for d, a in queries:
        if d == 'U':
            current_h -= 1
        elif d == 'D':
            current_h += 1
        elif d == 'L':
            current_w -= 1
        elif d == 'R':
            current_w += 1
        
        # Update the grid
        A[current_h][current_w] = a
        
        # Recompute dp from (current_h, current_w) to (H-1, W-1)
        for h in range(current_h, H):
            for w in range(current_w, W):
                if h == 0 and w == 0:
                    dp[h][w] = A[h][w]
                else:
                    dp[h][w] = 0
                    if h > 0:
                        dp[h][w] += dp[h-1][w]
                    if w > 0:
                        dp[h][w] += dp[h][w-1]
                    dp[h][w] = (dp[h][w] * A[h][w]) % MOD
        
        # Output the result for this query
        print(dp[H-1][W-1])

main()