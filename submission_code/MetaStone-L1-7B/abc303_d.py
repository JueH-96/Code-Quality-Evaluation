import sys
import math

def main():
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    n = len(S)
    
    INF = math.inf
    dp = [[INF] * 2 for _ in range(n + 1)]
    dp[0][0] = 0  # Starting with Caps Lock off
    
    for i in range(n):
        for s in [0, 1]:
            if dp[i][s] == INF:
                continue
            
            # Action 1: type 'a' or 'A'
            if (s == 0 and S[i] == 'a') or (s == 1 and S[i] == 'A'):
                if dp[i][s] + X < dp[i+1][s]:
                    dp[i+1][s] = dp[i][s] + X
            
            # Action 2: type 'A' or 'a'
            if (s == 0 and S[i] == 'A') or (s == 1 and S[i] == 'a'):
                if dp[i][s] + Y < dp[i+1][s]:
                    dp[i+1][s] = dp[i][s] + Y
            
            # Action 3: toggle Caps Lock
            new_s = 1 - s
            if dp[i][s] + Z < dp[i+1][new_s]:
                dp[i+1][new_s] = dp[i][s] + Z
    
    result = min(dp[n][0], dp[n][1])
    print(result)

if __name__ == "__main__":
    main()