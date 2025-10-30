import sys
import copy

def main():
    import sys
    import copy

    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        C = line[0]
        P = line[1]
        S = line[2:2+P]
        wheels.append((C, P, S))

    # Initialize dp array
    dp = [0.0] * (M + 1)
    for i in range(M):
        dp[i] = 1e9  # A large value to represent initial state

    epsilon = 1e-10
    max_iterations = 10000
    for _ in range(max_iterations):
        old_dp = dp[:]
        for i in range(M):
            min_cost = float('inf')
            for j in range(N):
                C = wheels[j][0]
                P = wheels[j][1]
                S = wheels[j][2]
                sum_dp = 0.0
                for k in range(P):
                    next_i = i + S[k]
                    if next_i >= M:
                        sum_dp += 0.0
                    else:
                        sum_dp += dp[next_i]
                E_j = C + (1.0 / P) * sum_dp
                if E_j < min_cost:
                    min_cost = E_j
            dp[i] = min_cost
        # Check for convergence
        max_diff = max(abs(dp[i] - old_dp[i]) for i in range(M))
        if max_diff < epsilon:
            break
    print(dp[0])

if __name__ == "__main__":
    main()