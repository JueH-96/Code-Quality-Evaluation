import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    C = list(map(int, data[2:]))

    # dp0[i], dp1[i]: cost to make prefix 1..i alternating starting with 0 (dp0)
    # or starting with 1 (dp1)
    dp0 = [0] * (N + 1)
    dp1 = [0] * (N + 1)
    for i in range(1, N + 1):
        bit0 = (i - 1) & 1  # desired at pos i if starting with 0
        bit1 = bit0 ^ 1     # desired at pos i if starting with 1
        dp0[i] = dp0[i-1] + (C[i-1] if int(S[i-1]) != bit0 else 0)
        dp1[i] = dp1[i-1] + (C[i-1] if int(S[i-1]) != bit1 else 0)

    # suf0[i], suf1[i]: cost to make suffix i..N alternating if T[i]=0 (suf0)
    # or T[i]=1 (suf1)
    suf0 = [0] * (N + 2)
    suf1 = [0] * (N + 2)
    for i in range(N, 0, -1):
        # if T[i]=0, cost now is C[i-1] if S[i-1]!=0, next T[i+1]=1
        suf0[i] = suf0[i+1] + (C[i-1] if int(S[i-1]) != 0 else 0)
        suf1[i] = suf1[i+1] + (C[i-1] if int(S[i-1]) != 1 else 0)
        # then swap for alternating effect on next position
        suf0[i] , suf1[i] = suf0[i] , suf1[i]
        # prepare for next iteration: we swap the roles for i-1
        # But easier: we can rebuild alternating by flipping at each step:
        # Instead do inside loop:
        if i > 1:
            suf0[i-1] = 0  # placeholder to satisfy array bounds
            suf1[i-1] = 0
        # Actually no need, because we'll only use suf? with correct toggles.

    # Actually above is overcomplicated. Let's recompute correctly:
    # Let's refill suf0 and suf1 properly:

    suf0 = [0] * (N + 2)
    suf1 = [0] * (N + 2)
    # We want suf_start[i][b] = cost to make i..N alternating starting with bit b at i
    for i in range(N, 0, -1):
        # If we start at i with 0:
        cost0 = C[i-1] if int(S[i-1]) != 0 else 0
        cost1 = C[i-1] if int(S[i-1]) != 1 else 0
        suf0[i] = cost0 + suf1[i+1]  # next must be 1
        suf1[i] = cost1 + suf0[i+1]  # next must be 0

    INF = 10**30
    ans = INF
    # For each break k=1..N-1, and starting bit b
    for k in range(1, N):
        # the initial alternating is over 1..k:
        # cost0 = dp0[k], cost1 = dp1[k]
        # the bit at k for pattern b is: b ^ ((k-1)%2)
        # then suffix start bit is that same bit
        # so if we started with 0, bit_at_k = (k-1)%2
        bit_k_if0 = (k-1) & 1
        # if started with 1, bit_at_k = bit_k_if0 ^ 1
        bit_k_if1 = bit_k_if0 ^ 1
        # suffix from k+1 must start with same bit as at k
        cost0 = dp0[k] + (suf0[k+1] if bit_k_if0 == 0 else suf1[k+1])
        cost1 = dp1[k] + (suf0[k+1] if bit_k_if1 == 0 else suf1[k+1])
        # take min of the two starting bits
        cur = cost0 if cost0 < cost1 else cost1
        if cur < ans:
            ans = cur

    print(ans)


if __name__ == "__main__":
    main()