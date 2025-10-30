import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We compute prefix-xors: px[0] = 0, px[i] = A[1] ^ ... ^ A[i].
    # The required sum is
    #   sum_{1<=i<j<=N} (A_i ^ ... ^ A_j)
    # = sum_{0<=l<r<=N} (px[l] ^ px[r])   minus the subarrays of length 1
    # = [ sum of all prefix-pair XORs ] - sum(A_i).
    #
    # sum_{0<=l<r<=N}(px[l]^px[r]) can be done bitwise:
    # For each bit k, let c = count of prefixes with bit k = 1, M=N+1.
    # There are c*(M-c) pairs that differ at bit k, each contributes (1<<k).
    # Then subtract sum of A's bits (because we must remove adjacent pairs l, l+1).

    # Count for prefix-xors
    maxb = 30  # enough for A[i] up to 1e8
    cnt_px = [0]*(maxb+1)
    px = 0
    # px[0] = 0 contributes no 1-bits
    for x in A:
        px ^= x
        # count bits in the new prefix
        for b in range(maxb+1):
            if px & (1<<b):
                cnt_px[b] += 1

    M = N + 1
    # count bits in A itself to subtract length-1 subarrays
    cnt_a = [0]*(maxb+1)
    for x in A:
        for b in range(maxb+1):
            if x & (1<<b):
                cnt_a[b] += 1

    ans = 0
    for b in range(maxb+1):
        c = cnt_px[b]
        # total differing pairs among prefixes
        total_pairs = c * (M - c)
        # subtract the single-element subarrays
        total_pairs -= cnt_a[b]
        ans += total_pairs * (1<<b)

    print(ans)

if __name__ == "__main__":
    main()