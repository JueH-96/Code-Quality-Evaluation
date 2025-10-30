import sys
import math

def solve():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = []
    idx = 1
    for _ in range(N):
        x = float(input_data[idx]); y = float(input_data[idx+1])
        coords.append((x,y))
        idx += 2
    
    # If N=2, there's no chance to skip any intermediate checkpoint (since there is none).
    # The answer is simply the distance from checkpoint 1 to checkpoint N with penalty=0.
    if N == 2:
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        ans = math.hypot(dx, dy)
        print(f"{ans:.9f}")
        return
    
    # Because penalty = 0 if we skip 0 checkpoints, or 2^(C - 1) if we skip C>0,
    # large numbers of skips quickly become expensive.  We will cap the maximum skips
    # at CMAX (for example 30), since 2^29 = 536870912 which typically outstrips
    # any possible distance advantage for realistic coordinates up to 10^4.
    CMAX = 30
    
    # Precompute penalty for c in [0..CMAX]
    # penalty[0]=0, penalty[c]=2^(c-1) for c>=1
    penalty = [0]*(CMAX+1)
    for c in range(1, CMAX+1):
        penalty[c] = 2**(c-1)
    
    INF = float('inf')
    
    # dp[i][c] = minimum distance to reach checkpoint i (0-based) having skipped c total
    #            of the intermediate checkpoints so far.
    dp = [[INF]*(CMAX+1) for _ in range(N)]
    dp[0][0] = 0.0  # Start at checkpoint 0 with 0 distance and 0 skips
    
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    
    for i in range(N):
        for c in range(CMAX+1):
            dist_i = dp[i][c]
            if dist_i == INF:
                continue
            # We can jump from i to k > i. We skip all checkpoints i+1..k-1.
            # The number of newly skipped points is (k - i - 1).
            # So c_new = c + (k - i - 1).
            # We must ensure c_new <= CMAX.
            # Also, to avoid large loops, we only extend up to i+ (CMAX - c) + 1.
            skip_max = CMAX - c
            max_k = min(N-1, i + skip_max + 1)
            
            xi, yi = x[i], y[i]
            for k in range(i+1, max_k+1):
                c_new = c + (k - i - 1)
                dx = x[k] - xi
                dy = y[k] - yi
                new_dist = dist_i + math.hypot(dx, dy)
                if new_dist < dp[k][c_new]:
                    dp[k][c_new] = new_dist
    
    # The answer is min over c in [0..CMAX] of dp[N-1][c] + penalty[c]
    ans = INF
    for c in range(CMAX+1):
        val = dp[N-1][c] + penalty[c]
        if val < ans:
            ans = val
    
    # Print with an absolute or relative error up to 1e-5
    print(f"{ans:.9f}")

def main():
    solve()

if __name__ == "__main__":
    main()