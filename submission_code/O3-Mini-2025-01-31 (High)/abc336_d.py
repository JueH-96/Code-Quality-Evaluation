# YOUR CODE HERE
def main():
    import sys
    from collections import deque
    # Read input (we use sys.stdin.buffer.read for speed)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    try:
        n = int(data[0])
    except:
        return
    # Read A as a list of integers.
    A = [int(x) for x in data[1:]]
    N = n

    # Precompute two auxiliary arrays.
    # L2[i] = A[i] - i   and   R2[i] = A[i] + i.
    L2 = [A[i] - i for i in range(N)]
    R2 = [A[i] + i for i in range(N)]

    # A helper function returning the sliding minimum for window size w from an array.
    def sliding_min(arr, w):
        # Returns a list "res" of length (len(arr) - w + 1) where:
        #   res[i] = min(arr[i], arr[i+1], ..., arr[i+w-1])
        dq = deque()
        res = [0] * (len(arr) - w + 1)
        for i, val in enumerate(arr):
            while dq and arr[dq[-1]] >= val:
                dq.pop()
            dq.append(i)
            if dq[0] <= i - w:
                dq.popleft()
            if i >= w - 1:
                res[i - w + 1] = arr[dq[0]]
        return res

    # Given a candidate k, check whether some contiguous block of length 2*k-1 can be made into a pyramid.
    def can_achieve(k):
        L = 2 * k - 1
        if L > N:
            return False
        # M is the number of possible starting indices for a block
        M = N - L + 1   # i goes from 0 to M-1.
        # For the left half (positions 0,...,k-1 in the block):
        # We need:   min_{x in [i, i+k-1]} (A[x] - (x-i)) >= 1  <=>  min_{x in [i,i+k-1]} L2[x] >= 1 - i.
        LM = sliding_min(L2, k)  # LM[i] = min(L2[i],..., L2[i+k-1])
        # For the right half (if k>1) we do a similar sliding–window on R2.
        if k > 1:
            # For positions x in [i+k, i+2*k-2] we need:
            #    min{ A[x] + (x-i) } >= 2*k-1   <=>  min{ R2[x] } >= 2*k-1 + i.
            rm = sliding_min(R2, k - 1)  # rm[j] = min(R2[j],..., R2[j+(k-1)-1])
        # Check every possible starting index i.
        for i in range(M):
            if LM[i] < 1 - i:
                continue
            if k > 1:
                # The right–half window for starting index i is [i+k, i+2*k-2].
                # The sliding_min for R2 returns rm; its entry at index (i+k) is the minimum over that window.
                if rm[i + k] < (2 * k - 1 + i):
                    continue
            return True
        return False

    # Binary search for the maximum k. (The trivial pyramid of size 1 is always achievable.)
    lo, hi = 1, (N + 1) // 2
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_achieve(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()