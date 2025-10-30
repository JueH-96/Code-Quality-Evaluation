import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    wheels = []
    for _ in range(N):
        C_i = int(next(it))
        P_i = int(next(it))
        outcomes = [int(next(it)) for _ in range(P_i)]
        n0_i = outcomes.count(0)
        wheels.append((C_i, P_i, n0_i, outcomes))
    
    dp = [0.0] * (M + 1)
    
    for x in range(M - 1, -1, -1):
        best = float('inf')
        for wheel in wheels:
            C_i, P_i, n0_i, outcomes = wheel
            s_sum = 0.0
            for s in outcomes:
                if s == 0:
                    continue
                next_state = x + s
                if next_state < M:
                    s_sum += dp[next_state]
            denom = P_i - n0_i
            candidate = (C_i * P_i + s_sum) / denom
            if candidate < best:
                best = candidate
        dp[x] = best
    
    print("{:.12f}".format(dp[0]))

if __name__ == "__main__":
    main()