import sys
import math

def main():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, sys.stdin.readline().split())
        wheels.append((C, P, S))
    
    # Precompute the expected points and cost for each wheel
    expected = []
    for C, P, S in wheels:
        total = sum(S)
        avg = total / P
        expected.append((C, avg))
    
    # Dynamic programming to find the minimal expected cost
    # dp[m] represents the minimal expected cost to reach at least m points
    dp = [float('inf')] * (M + 1)
    dp[0] = 0
    for m in range(M):
        if dp[m] == float('inf'):
            continue
        for C, avg in expected:
            new_m = min(m + avg, M)
            dp[new_m] = min(dp[new_m], dp[m] + C)
    
    # Since the expected value is the minimal cost to reach at least M points
    # and the dp array is filled with the minimal cost for each possible point
    # the answer is dp[M]
    print("{0:.20f}".format(dp[M]))

if __name__ == "__main__":
    main()