# YOUR CODE HERE
import sys
from typing import List, Tuple
import heapq

def read_input() -> Tuple[int, int, List[Tuple[int, List[int]]]]:
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        line = list(map(int, input().split()))
        C, P = line[:2]
        S = line[2:]
        wheels.append((C, S))
    return N, M, wheels

def expected_cost(M: int, wheels: List[Tuple[int, List[int]]]) -> float:
    dp = [float('inf')] * (M + 1)
    dp[0] = 0
    
    for points in range(M):
        for cost, outcomes in wheels:
            exp_gain = sum(outcomes) / len(outcomes)
            exp_cost = cost / exp_gain
            
            new_cost = dp[points] + exp_cost
            for outcome in outcomes:
                next_points = min(M, points + outcome)
                dp[next_points] = min(dp[next_points], new_cost)
    
    return dp[M]

def main():
    N, M, wheels = read_input()
    result = expected_cost(M, wheels)
    print(f"{result:.15f}")

if __name__ == "__main__":
    main()