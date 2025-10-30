# YOUR CODE HERE
import sys

def solve():
    N, W = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    
    dp = [0] * (W + 1)
    
    for w, v in items:
        for i in range(W, w - 1, -1):
            j = 0
            while True:
                if j * w > i:
                    break
                dp[i] = max(dp[i], dp[i - j * w] + j * v - j * j)
                j += 1
    
    print(dp[W])

if __name__ == "__main__":
    solve()