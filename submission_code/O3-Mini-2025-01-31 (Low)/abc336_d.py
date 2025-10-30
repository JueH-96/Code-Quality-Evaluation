def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # It is always allowed to decrease any value and remove first/last entries.
    # In order to “morph” some contiguous segment of A into a pyramid sequence
    # with size k (which has length L = 2k-1), note that after allowed operations we
    # can only lower values. Therefore, for each position in the candidate contiguous segment,
    # we must have had A[x] at least as large as the target pyramid value.
    #
    # A pyramid (of size k) is: 1, 2, …, k-1, k, k-1, …, 2, 1.
    # Writing indices 0-based, the required value at position j in the segment
    # is: req(j) = min(j+1, L - j)  where L = 2k-1.
    #
    # Since we can only reduce A’s values (never increase) we need:
    #    For any contiguous segment (of length L) starting at index i,
    #    for each j,  A[i+j] >= req(j).
    #
    # Our task is now to choose a contiguous segment (obtained by removals)
    # and then decrement some entries so that for each j, the final value equals req(j).
    # The only “limiting” condition is that for each j,
    #    A[i+j] must be at least req(j).
    #
    # We want the maximum k for which there exists an i such that for all j in 0..L-1:
    #    A[i+j] >= min(j+1, L - j).
    #
    # We split the conditions into two halves:
    #
    #  1) For the increasing part j in [0, k-1]: req(j) = j+1, so
    #         A[i+j] >= j+1.
    #     Rearranging, A[i+j] - j >= 1.
    #
    #  2) For the decreasing part j in [k-1, 2k-2]:
    #     Let t = j - (k-1), so t in [0, k-1]. Then req(j) = (2k-1) - j
    #         = 2k-1 - (t+k-1) = k - t.
    #     That is, A[i+j] >= k - t.
    #     Rearranging, A[i+j] + t >= k.
    #
    # To express these conditions in a form amenable to efficient range queries,
    # we shift indices to “absorb” the offset i.
    #
    # For the increasing part:
    #   Write j = x - i  for x in [i, i+k-1].
    #   Then the condition becomes:
    #         A[x] - (x - i) = (A[x] - x) + i >= 1.
    #   Define U[x] = A[x] - x.
    #   Then we need:  i + min_{x in [i, i+k-1]} U[x] >= 1.
    #
    # For the decreasing part:
    #   Write x = i+j for j in [k-1, 2k-2]. Let t = x - (i+k-1) so that t in [0, k-1].
    #   The condition A[x] + t >= k becomes:
    #         A[x] + (x - i - (k-1)) >= k
    #         A[x] + x >= i + 2k - 1.
    #   Define V[x] = A[x] + x.
    #   Then the condition is:  min_{x in [i+k-1, i+2*k-2]} V[x] >= i + 2*k - 1.
    #
    # Thus for a candidate contiguous segment starting at index i of length L = 2k-1,
    # we need both:
    #    (i)   i + (minimum of U[x] over x in [i, i+k-1]) >= 1,
    #   (ii)   (minimum of V[x] over x in [i+k-1, i+2*k-2]) >= i + 2*k - 1.
    #
    # Our plan is to binary search for the maximum k (1 <= k <= (n+1)//2)
    # such that there is some contiguous segment of length 2k-1 satisfying these conditions.
    #
    # To do range minimum queries quickly we build two sparse tables:
    # one for U and one for V.
    
    N = n
    U = [A[i] - i for i in range(N)]
    V = [A[i] + i for i in range(N)]
    
    # Build a sparse table for U and V so we can answer RMQs in O(1)
    log = [0]*(N+1)
    for i in range(2, N+1):
        log[i] = log[i//2] + 1
    kmax = log[N] + 1
    stU = [U[:]]
    stV = [V[:]]
    
    j = 1
    while (1 << j) <= N:
        prevU = stU[j - 1]
        prevV = stV[j - 1]
        currU = []
        currV = []
        length = 1 << (j - 1)
        for i in range(N - (1 << j) + 1):
            # We compute the minimum from i to i+2^j -1 for U and V.
            if prevU[i] <= prevU[i + length]:
                currU.append(prevU[i])
            else:
                currU.append(prevU[i + length])
            if prevV[i] <= prevV[i + length]:
                currV.append(prevV[i])
            else:
                currV.append(prevV[i + length])
        stU.append(currU)
        stV.append(currV)
        j += 1

    def query(st, L, R):
        # Returns the minimum value in st over the interval [L, R].
        j = log[R - L + 1]
        cand1 = st[j][L]
        cand2 = st[j][R - (1 << j) + 1]
        return cand1 if cand1 <= cand2 else cand2

    # Helper routine: given k (pyramid size), check if we can find some contiguous segment of length L=2*k-1
    # satisfying the two conditions.
    def can(k):
        L_seg = 2 * k - 1
        if L_seg > n:
            return False
        # We can only choose i from 0 to n - L_seg.
        for i in range(0, n - L_seg + 1):
            # Condition (i): for indices [i, i+k-1] in U.
            minU = query(stU, i, i + k - 1)
            if i + minU < 1:
                continue
            # Condition (ii): for indices [i+k-1, i+2*k-2] in V.
            minV = query(stV, i + k - 1, i + 2 * k - 2)
            if minV < i + 2 * k - 1:
                continue
            return True
        return False

    # Binary search for the maximum k that works.
    lo = 1
    hi = (n + 1) // 2
    ans = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if can(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()