import sys
import math

def main():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, sys.stdin.readline().split())
        wheels.append((C, P, S))
    
    # Initialize DP array
    dp = [float('inf')] * (M + 1)
    dp[0] = 0
    
    for points in range(M):
        if dp[points] == float('inf'):
            continue
        for C, P, S in wheels:
            expected_cost = 0
            for s in S:
                new_points = min(points + s, M)
                expected_cost += dp[new_points]
            expected_cost = C + (expected_cost / P)
            if expected_cost < dp[points]:
                dp[points] = expected_cost
    
    print("{0:.20f}".format(dp[0]))

if __name__ == "__main__":
    main()