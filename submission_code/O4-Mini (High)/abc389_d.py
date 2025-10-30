def main():
    import sys
    from math import isqrt

    data = sys.stdin.readline().strip()
    if not data:
        return
    R = int(data)

    # We want integer (i,j) so that the farthest corner distance
    # sqrt((|i|+0.5)^2 + (|j|+0.5)^2) ≤ R.
    # Let M = R-1 (since |i| ≤ R-0.5 ⇒ |i| ≤ R-1 for integer i).
    # We loop i = 0..R-1, count j for nonnegative i, and reflect across axes.
    #
    # For fixed i ≥ 0, define
    #   K = R^2 - (i+0.5)^2
    # We need (|j|+0.5)^2 ≤ K ⇒ |j| ≤ sqrt(K) - 0.5.
    # Equivalently, let S = 4*K = 4*R^2 - (2*i+1)^2.
    # We find the largest integer m ≥ 0 with (2m+1)^2 ≤ S.
    # Then j runs from -m..m, i.e. total 2*m+1 values.
    #
    # Finally, i=0 contributes once; i>0 contributes twice (±i).
    #
    # This is O(R) with a fast integer isqrt.

    R2 = R * R
    total4R2 = 4 * R2
    ans = 0

    for i in range(R):  # i = 0..R-1
        v = 2 * i + 1
        S = total4R2 - v * v
        # S >= 0 for i <= R-1
        T = isqrt(S)        # floor(sqrt(S))
        m = (T - 1) // 2    # max m with (2m+1) <= T
        cnt_j = 2 * m + 1   # j = -m..m

        # add contributions, reflect i>0
        if i == 0:
            ans += cnt_j
        else:
            ans += cnt_j * 2

    print(ans)

main()