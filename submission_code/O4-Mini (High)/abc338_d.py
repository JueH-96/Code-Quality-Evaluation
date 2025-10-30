import sys
def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    # We'll compute F(c) = sum of distances when cutting bridge (c, c+1).
    # It can be written as total_d1 + S(c), where total_d1 = sum of clockwise distances d1,
    # and S(c) = sum over k of (d2_k - d1_k) for all k such that c lies in the clockwise arc [X_k, X_{k+1}).
    # We let delta_k = d2_k - d1_k = N - 2*d1_k, and we add delta_k to all positions c in that interval.
    # We do a range‐add with a difference array over 1..N (mod intervals).
    
    D = [0] * (N + 2)
    total_d1 = 0
    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]
        # clockwise distance from a to b
        d1 = b - a
        if d1 < 0:
            d1 += N
        # accumulate the base cost
        total_d1 += d1
        # delta for positions where the clockwise arc is broken
        delta = N - 2 * d1
        # interval I = [a .. b) mod N
        if a < b:
            # no wrap: [a..b-1]
            D[a] += delta
            D[b] -= delta
        else:
            # wrap-around: [a..N] and [1..b-1]
            D[a] += delta
            D[N + 1] -= delta
            if b > 1:
                D[1] += delta
                D[b] -= delta

    # build prefix sums to get S(c) for c=1..N, track minimum
    cur = 0
    min_S = float('inf')
    for c in range(1, N + 1):
        cur += D[c]
        if cur < min_S:
            min_S = cur

    # answer is base + minimum adjustment
    ans = total_d1 + min_S
    print(ans)

# call main
main()