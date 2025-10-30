import sys
import math

def main():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    
    INF = float('-inf')
    dp = [INF] * (N + 1)
    dp[0] = 0.0
    current_max = 0
    
    for p in P:
        # Iterate from current_max down to 0 to prevent overwriting
        for m in range(current_max, -1, -1):
            if dp[m] != INF:
                new_m = m + 1
                new_val = dp[m] * 0.9 + p
                if new_val > dp[new_m]:
                    dp[new_m] = new_val
        current_max += 1
    
    max_R = -float('inf')
    for m in range(1, N + 1):
        if dp[m] == INF:
            continue
        D = 10.0 * (1.0 - (0.9 ** m))
        term1 = dp[m] / D
        term2 = 1200.0 / math.sqrt(m)
        R = term1 - term2
        if R > max_R:
            max_R = R
    
    print("{0:.10f}".format(max_R))

if __name__ == "__main__":
    main()