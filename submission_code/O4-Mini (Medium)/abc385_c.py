def main():
    import sys
    from array import array
    input = sys.stdin.readline

    N = int(input())
    H = list(map(int, input().split()))

    # Group positions (0-based) by height
    pos_by_h = {}
    for idx, h in enumerate(H):
        pos_by_h.setdefault(h, []).append(idx)

    # Sort groups by descending size to raise ans quickly and prune
    groups = sorted(pos_by_h.values(), key=lambda lst: -len(lst))

    ans = 1
    # Auxiliary index array for fast lookup of position -> rank in pos list
    pos_idx = [-1] * N

    for pos in groups:
        m = len(pos)
        # If this group can't beat current ans, break
        if m <= ans:
            break

        # build pos_idx for this group
        for rank, p in enumerate(pos):
            pos_idx[p] = rank

        # dp[k] will be an array of unsigned shorts of length k:
        # dp[k][j] = length of AP ending at pos[j], pos[k], for j<k
        dp = []  # list of array('H')

        # iterate k from 0..m-1
        for k in range(m):
            # initialize dp_k with 2s
            # type 'H' is unsigned short sufficient for lengths <= 3000
            dp_k = array('H', [2]) * k
            pk = pos[k]
            # for each j<k compute dp[j][k]
            for j in range(k):
                pj = pos[j]
                # target previous term in AP: pi = 2*pj - pk
                ti = pj * 2 - pk
                if 0 <= ti < N:
                    i = pos_idx[ti]
                    if i >= 0:
                        # extend existing AP ending at (i, j)
                        v = dp[j][i] + 1
                    else:
                        v = 2
                else:
                    v = 2
                dp_k[j] = v
                if v > ans:
                    ans = v
            dp.append(dp_k)

        # cleanup pos_idx for this group
        for p in pos:
            pos_idx[p] = -1

    # print result
    print(ans)

if __name__ == "__main__":
    main()