import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C_i = parts[0]
        P_i = parts[1]
        S_list = parts[2:]
        wheels.append((C_i, P_i, S_list))
    
    # Initialize dp array
    dp = [0.0] * (M + 1)  # dp[x] for x >= M is 0
    
    for x in range(M-1, -1, -1):
        min_expected = float('inf')
        for (C_i, P_i, S_list) in wheels:
            sum_val = 0.0
            count = 0
            for s in S_list:
                next_p = x + s
                if next_p >= M:
                    pass  # contribute 0
                else:
                    if next_p == x:
                        count += 1
                    else:
                        sum_val += dp[next_p]
            numerator = C_i * P_i + sum_val
            denominator = P_i - count
            candidate = numerator / denominator
            if candidate < min_expected:
                min_expected = candidate
        dp[x] = min_expected
    
    print(dp[0])

if __name__ == "__main__":
    main()