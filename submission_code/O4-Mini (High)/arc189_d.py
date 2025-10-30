import sys
def main():
    import sys
    from array import array
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    # build log table
    lg = [0] * (N+1)
    for i in range(2, N+1):
        lg[i] = lg[i>>1] + 1
    maxlog = lg[N]
    # 2^k table
    B2 = [1<<k for k in range(maxlog+1)]

    # build sparse‐table fwd[k][i] = max of A[i..i+2^k-1]
    fwd = [None] * (maxlog+1)
    fwd[0] = array('I', A)
    for k in range(1, maxlog+1):
        prev = fwd[k-1]
        shift = B2[k-1]
        length = N - B2[k] + 1
        curr = array('I', [0]) * length
        for i in range(length):
            x = prev[i]
            y = prev[i+shift]
            curr[i] = x if x >= y else y
        fwd[k] = curr

    # build sparse‐table over reversed A
    revA = A[::-1]
    fwd_rev = [None] * (maxlog+1)
    fwd_rev[0] = array('I', revA)
    for k in range(1, maxlog+1):
        prev = fwd_rev[k-1]
        shift = B2[k-1]
        length = N - B2[k] + 1
        curr = array('I', [0]) * length
        for i in range(length):
            x = prev[i]
            y = prev[i+shift]
            curr[i] = x if x >= y else y
        fwd_rev[k] = curr

    # prefix sums
    P = [0] * (N+1)
    s = 0
    for i, v in enumerate(A):
        s += v
        P[i+1] = s

    B = [0]*N
    # local refs for speed
    A0 = A; P0 = P; lg0 = lg; b2 = B2
    fwd0 = fwd; fwdrev0 = fwd_rev; N0 = N

    for K in range(N0):
        Ak = A0[K]
        # quick skip if no neighbor is strictly smaller
        left0 = (K>0 and A0[K-1] < Ak)
        right0 = (K<N0-1 and A0[K+1] < Ak)
        if not (left0 or right0):
            B[K] = Ak
            continue

        # simulate greedy block‐expansion
        S = Ak
        l = K
        r = K
        while True:
            canL = (l>0 and A0[l-1] < S)
            canR = (r<N0-1 and A0[r+1] < S)
            if not (canL or canR):
                break
            # choose direction
            if not canL:
                direc = 1  # right
            elif not canR:
                direc = 0  # left
            else:
                # tie: absorb the smaller neighbor first
                if A0[l-1] <= A0[r+1]:
                    direc = 0
                else:
                    direc = 1

            if direc == 1:
                # expand right
                lo = r+1
                if lo >= N0:
                    newR = N0 - 1
                else:
                    rr = N0 - 1
                    length = rr - lo + 1
                    k = lg0[length]
                    row = fwd0[k]
                    # max on [lo..rr]
                    m1 = row[lo]
                    m2 = row[rr - b2[k] + 1]
                    if m1 < S and m2 < S:
                        newR = rr
                    else:
                        # binary‐search the first barrier ≥ S
                        lb = lo; ub = rr; ans = rr
                        while lb <= ub:
                            mid = (lb + ub) >> 1
                            length2 = mid - lo + 1
                            k2 = lg0[length2]
                            row2 = fwd0[k2]
                            t1 = row2[lo]
                            t2 = row2[mid - b2[k2] + 1]
                            if t1 >= S or t2 >= S:
                                ans = mid
                                ub = mid - 1
                            else:
                                lb = mid + 1
                        newR = ans - 1
                # absorb [r+1..newR]
                added = P0[newR+1] - P0[r+1]
                S += added
                r = newR
            else:
                # expand left via reversed array
                lo_rev = N0 - l
                if lo_rev >= N0:
                    newL = 0
                else:
                    rr = N0 - 1
                    length = rr - lo_rev + 1
                    k = lg0[length]
                    row = fwdrev0[k]
                    m1 = row[lo_rev]
                    m2 = row[rr - b2[k] + 1]
                    if m1 < S and m2 < S:
                        newL = 0
                    else:
                        lb = lo_rev; ub = rr; ans = rr
                        while lb <= ub:
                            mid = (lb + ub) >> 1
                            length2 = mid - lo_rev + 1
                            k2 = lg0[length2]
                            row2 = fwdrev0[k2]
                            t1 = row2[lo_rev]
                            t2 = row2[mid - b2[k2] + 1]
                            if t1 >= S or t2 >= S:
                                ans = mid
                                ub = mid - 1
                            else:
                                lb = mid + 1
                        # map back to original index
                        newL = N0 - ans
                added = P0[l] - P0[newL]
                S += added
                l = newL

        B[K] = S

    # output
    print(" ".join(map(str, B)))


if __name__ == "__main__":
    main()