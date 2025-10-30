import sys
import threading

def main():
    import sys, math
    data = sys.stdin.read().split()
    N = int(data[0])
    P = list(map(int, data[1:]))
    # Reverse P so that weights go alpha^0 for first selected, alpha^1 for second, ...
    Q = P[::-1]
    alpha = 0.9

    # Precompute powers of alpha: alpha_pows[k] = alpha^k
    alpha_pows = [1.0] * (N + 1)
    for i in range(1, N+1):
        alpha_pows[i] = alpha_pows[i-1] * alpha

    # Precompute prefix sums of weights S[k] = sum_{r=0..k-1} alpha^r
    S = [0.0] * (N + 1)
    # S[0] = 0
    for k in range(1, N+1):
        S[k] = S[k-1] + alpha_pows[k-1]

    # dp[j] = maximum weighted sum picking j contests so far
    # initialize dp[0] = 0, others = -inf
    neg_inf = float('-inf')
    dp = [neg_inf] * (N+1)
    dp[0] = 0.0

    # Fill DP: for each contest in reversed order
    for t in range(1, N+1):
        x = Q[t-1]
        # Update dp backwards to avoid overwrite
        # when we pick x as the j-th selected, it contributes alpha^{j-1} * x
        # so dp_new[j] = max(dp_old[j], dp_old[j-1] + alpha_pows[j-1]*x)
        # we do it in-place backwards
        # local references for speed
        dp_jm1 = dp  # just alias
        ap = alpha_pows
        for j in range(t, 0, -1):
            # candidate if we take this as the j-th pick
            cand = dp_jm1[j-1] + ap[j-1] * x
            if cand > dp_jm1[j]:
                dp_jm1[j] = cand

    # Now dp[k] holds the best weighted sum for selecting k contests
    # Compute the maximum rating R_k = dp[k]/S[k] - 1200/sqrt(k)
    best = -1e300
    for k in range(1, N+1):
        val = dp[k]/S[k] - 1200.0/math.sqrt(k)
        if val > best:
            best = val

    # Print with enough precision
    sys.stdout.write("{:.15f}
".format(best))


if __name__ == "__main__":
    main()