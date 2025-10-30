import math
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = []
    Y = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        X.append(x)
        Y.append(y)
        idx += 2
    
    cap = 20
    INF = 1e18
    dp = [[INF] * (cap + 1) for _ in range(N)]
    dp[0][0] = 0.0  # start at checkpoint 0, 0 skipped so far
    
    for i in range(1, N):
        start_j = max(0, i - (cap + 2))
        for j in range(start_j, i):
            for k_prev in range(cap + 1):
                if dp[j][k_prev] == INF:
                    continue
                skipped = i - j - 1
                new_k = k_prev + skipped
                if new_k > cap:
                    continue
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                dist = math.hypot(dx, dy)
                if dp[i][new_k] > dp[j][k_prev] + dist:
                    dp[i][new_k] = dp[j][k_prev] + dist
    
    # Now calculate candidates
    candidates = []
    # Check all possibilities in dp[N-1]
    for k in range(cap + 1):
        if dp[N-1][k] == INF:
            continue
        C = k
        if C == 0:
            penalty = 0.0
        else:
            penalty = 2.0 ** (C - 1)
        total = dp[N-1][k] + penalty
        candidates.append(total)
    
    # Check direct path
    direct_dx = X[-1] - X[0]
    direct_dy = Y[-1] - Y[0]
    direct_dist = math.hypot(direct_dx, direct_dy)
    direct_C = N - 2
    if direct_C == 0:
        direct_penalty = 0.0
    else:
        direct_penalty = 2.0 ** (direct_C - 1)
    direct_total = direct_dist + direct_penalty
    candidates.append(direct_total)
    
    answer = min(candidates)
    print("{0:.20f}".format(answer))
    
if __name__ == '__main__':
    main()