def main():
    import sys
    import math

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    points = [(int(next(it)), int(next(it))) for _ in range(n)]
    
    # dp[i] stores the minimum cost to reach checkpoint (i+1)
    dp = [float('inf')] * n
    dp[0] = 0.0

    # A direct jump from checkpoint j to i (j < i) has:
    #   Euclidean distance = distance(points[j], points[i])
    #   And if we skip k = i-j-1 checkpoints,
    #       penalty = 0 if k==0 else 2^(k-1).
    #
    # However, although a naive O(n^2) solution would iterate over all j < i,
    # note that penalty grows exponentially with the number of skipped checkpoints.
    # This means that skipping too many checkpoints is never beneficial.
    # We can thereby limit the jump length. Experimentally, we can restrict to a
    # maximum jump length max_jump. Here we choose max_jump = 50 (so skip count up to 49)
    # which is more than enough given the scale of coordinates.
    
    max_jump = 50
    
    # Precompute penalty for skip count s=1..max_jump (s== number of skipped checkpoints)
    # For s > 0, penalty = 2^(s-1), and for s==0 it is 0.
    penalty = [0] * (max_jump + 1)
    for s in range(1, max_jump + 1):
        penalty[s] = 2 ** (s - 1)
    
    # DP over checkpoints: iterate for each destination checkpoint i
    for i in range(1, n):
        # Consider only previous checkpoint indices j that are not too far behind.
        j_min = max(0, i - (max_jump + 1))
        (xi, yi) = points[i]
        for j in range(j_min, i):
            (xj, yj) = points[j]
            # Euclidean distance between checkpoint j and i
            d = math.hypot(xi - xj, yi - yj)
            # Number of skipped checkpoints is (i - j - 1)
            skip = i - j - 1
            # Compute added penalty. If skip > max_jump, we directly compute it
            # but notice that due to exponential growth, such a transition is unlikely to be optimal.
            if skip == 0:
                pen = 0
            elif skip <= max_jump:
                pen = penalty[skip]
            else:
                pen = 2 ** (skip - 1)
            candidate = dp[j] + d + pen
            if candidate < dp[i]:
                dp[i] = candidate

    # Print answer with sufficient precision.
    # The problem requires an absolute or relative error at most 1e-5.
    sys.stdout.write("{:.11f}".format(dp[-1]))

if __name__ == '__main__':
    main()