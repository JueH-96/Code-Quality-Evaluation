import sys
from collections import defaultdict
import heapq

def read_input():
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        line = list(map(int, input().split()))
        cost = line[0]
        P = line[1]
        scores = line[2:]
        wheels.append((cost, scores))
    return N, M, wheels

def solve(N, M, wheels):
    # dp[points] = expected cost to reach M points starting from points
    dp = defaultdict(lambda: float('inf'))
    dp[M] = 0
    
    # Process points from M-1 down to 0
    for points in range(M-1, -1, -1):
        # Try each wheel
        for i in range(N):
            cost, scores = wheels[i]
            P = len(scores)
            
            # Calculate expected cost using this wheel
            exp_cost = cost # Cost to spin
            exp_points = 0
            for score in scores:
                next_points = min(M, points + score)
                exp_points += dp[next_points] / P
            
            dp[points] = min(dp[points], exp_cost + exp_points)
    
    return dp[0]

def main():
    N, M, wheels = read_input()
    result = solve(N, M, wheels)
    print(result)

if __name__ == "__main__":
    main()