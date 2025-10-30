def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # Read N, K
    N, K = map(int, input().split())

    # A helper to compute the exponent of 2 in n! (i.e., how many factors of 2 are in n!)
    # e2_factorial(n) = sum_{k>=1} floor(n / 2^k).
    def e2_factorial(n):
        e2 = 0
        p = 2
        while p <= n:
            e2 += n // p
            p <<= 1
        return e2

    # Compute exponent of 2 in S = (N*K)! / (K!)^N
    # e2S > 0  => S is even
    # e2S = 0  => S is odd
    e2S = e2_factorial(N * K) - N * e2_factorial(K)

    # For small N*K, we can compute S exactly; otherwise do a "near 0.5" approach
    # because S can become astronomically large for bigger N,K.
    if N * K <= 20:
        import math
        factNK = math.factorial(N * K)
        factK = math.factorial(K)
        denom = factK ** N
        S = factNK // denom  # exact number of good sequences

        # M = floor((S+1)/2).  If S is even => M = S//2, if S is odd => M=(S+1)//2
        if S % 2 == 0:
            M = S // 2
        else:
            M = (S + 1) // 2

        # Our fraction R = M / S   (this is the "rank fraction" in [0,1])
        R = M / S

    else:
        # We cannot compute S exactly if N*K is large, so we just need to know
        # if S is even or odd.  If even => median rank fraction = 0.5
        # If odd  => median rank fraction ~ 0.5 + a tiny epsilon.
        if e2S > 0:
            R = 0.5  # S even
        else:
            # S odd => the exact fraction is 0.5 + 1/(2S), but 1/(2S) is tiny.
            # Use a small epsilon smaller than 1/(N*K), e.g. 1e-7:
            R = 0.5 + 1e-7

    # Fenwick (Binary Indexed) Tree to store frequencies of each integer 1..N
    fenwicks = [0] * (N + 1)

    # We'll store K copies of each i in 1..N initially
    # Build Fenwicks in O(N) or O(N log N); for N=500 it's fine either way.
    c0 = [0] * (N + 1)
    for i in range(1, N + 1):
        c0[i] = K

    # Build fenwicks array
    for i in range(1, N + 1):
        fenwicks[i] += c0[i]
        j = i + (i & -i)
        if j <= N:
            fenwicks[j] += fenwicks[i]

    def fenwicks_sum(idx):
        s = 0
        while idx > 0:
            s += fenwicks[idx]
            idx -= (idx & -idx)
        return s

    def fenwicks_update(idx, delta):
        while idx <= N:
            fenwicks[idx] += delta
            idx += (idx & -idx)

    # We want "the smallest i such that fenwicks_sum(i)/sumVal >= R".
    # We do a simple binary search over i in [1..N].
    def findIndex(R, sumVal):
        low, high = 1, N
        while low < high:
            mid = (low + high) // 2
            val = fenwicks_sum(mid) / sumVal
            if val < R:
                low = mid + 1
            else:
                high = mid
        return low

    # Now we iteratively pick each of the N*K positions.
    # sumVal = how many total items remain
    # R = "which fraction of the remaining permutations" we want.
    # Then after picking i at each step, we update R via:
    #   newR = [ oldR - (countBelow(i)/sumVal ) ] / [ c[i]/sumVal ]
    # because we want the rank inside that block.

    seq = []
    sumVal = N * K
    for _ in range(N * K):
        i = findIndex(R, sumVal)     # which integer goes in the next position
        seq.append(i)
        si_1 = fenwicks_sum(i - 1)   # number of elements strictly below i
        ci = fenwicks_sum(i) - si_1  # how many of i remain (should be >=1)
        fenwicks_update(i, -1)       # use one occurrence of i
        oldSum = sumVal
        sumVal -= 1
        # Update R = (R * oldSum - si_1) / ci
        # Explanation: partialSum(i-1)/oldSum < R <= partialSum(i)/oldSum
        # => R*oldSum - si_1 in (0..ci], so new R in (0..1].
        R = (R * oldSum - si_1) / ci if ci > 0 else 0.0

    # Output the result
    print(' '.join(map(str, seq)))