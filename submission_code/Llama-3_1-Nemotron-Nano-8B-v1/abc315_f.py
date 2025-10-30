import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    coords = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        coords.append((x, y))
        idx +=2
    
    # Precompute distances between all pairs (but for this code, we'll compute on the fly)
    def dist(a, b):
        dx = coords[a][0] - coords[b][0]
        dy = coords[a][1] - coords[b][1]
        return math.hypot(dx, dy)
    
    max_skips = 30  # Heuristic to limit the number of skips considered
    INF = float('inf')
    dp = [ [INF] * (max_skips +1) for _ in range(N)]
    dp[0][0] = 0.0  # Starting at checkpoint 0 (1st checkpoint) with 0 skips
    
    for i in range(N):
        for k in range(max_skips +1):
            if dp[i][k] == INF:
                continue
            # Try to jump to j from i, up to 31 steps ahead
            for j in range(i+1, min(i+32, N)):
                skips = j - i -1
                new_k = k + skips
                if new_k > max_skips:
                    continue
                d = dist(i, j)
                if dp[j][new_k] > dp[i][k] + d:
                    dp[j][new_k] = dp[i][k] + d
    
    min_s = INF
    for k in range(max_skips +1):
        if dp[N-1][k] == INF:
            continue
        penalty = 0.0 if k ==0 else (2 ** (k-1))
        total = dp[N-1][k] + penalty
        if total < min_s:
            min_s = total
    print("{0:.20f}".format(min_s))

if __name__ == "__main__":
    main()