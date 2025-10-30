# YOUR CODE HERE
import sys

def main():
    N, W = map(int, sys.stdin.readline().split())
    items = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Initialize DP table
    dp = [ -1e18 ] * (W + 1)
    dp[0] = 0
    
    for w, v in items:
        new_dp = [ -1e18 ] * (W + 1)
        for k in range(0, W // w + 1):
            happiness = k * v - k * k
            for prev_w in range(W - k * w + 1):
                if dp[prev_w] != -1e18:
                    new_w = prev_w + k * w
                    if new_w <= W:
                        new_dp[new_w] = max(new_dp[new_w], dp[prev_w] + happiness)
        dp = new_dp
    
    max_happiness = max(dp)
    print(max_happiness)

if __name__ == "__main__":
    main()