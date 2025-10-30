import sys

def main():
    N = int(sys.stdin.readline())
    steps = []
    for _ in range(N):
        A, S = sys.stdin.readline().split()
        steps.append((int(A), S))
    
    INF = float('inf')
    # Initialize DP: prev_dp[L][R] is the minimal cost to reach (L, R) after 0 steps
    prev_dp = [[0] * 101 for _ in range(101)]
    
    for a, s in steps:
        new_dp = [[INF] * 101 for _ in range(101)]
        for prev_L in range(1, 101):
            for prev_R in range(1, 101):
                cost = prev_dp[prev_L][prev_R]
                if cost == INF:
                    continue
                if s == 'L':
                    new_L = a
                    new_R = prev_R
                    delta = abs(a - prev_L)
                    new_cost = cost + delta
                    if new_cost < new_dp[new_L][new_R]:
                        new_dp[new_L][new_R] = new_cost
                else:
                    new_L = prev_L
                    new_R = a
                    delta = abs(a - prev_R)
                    new_cost = cost + delta
                    if new_cost < new_dp[new_L][new_R]:
                        new_dp[new_L][new_R] = new_cost
        prev_dp = new_dp
    
    min_val = INF
    for L in range(1, 101):
        for R in range(1, 101):
            if prev_dp[L][R] < min_val:
                min_val = prev_dp[L][R]
    print(min_val)

if __name__ == "__main__":
    main()