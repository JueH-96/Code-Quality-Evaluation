def main():
    import sys
    from collections import defaultdict

    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # total steps in one full circle
    total = sum(A)
    T = total % M

    # We represent each rest area uniquely by a prefix sum.
    # Define P[0], P[1], ..., P[n-1] corresponding to rest areas 1,2,...,n.
    # Let P[0] = 0 (starting point at rest area 1), then for i from 1 to n-1:
    # P[i] = (A[0] + A[1] + ... + A[i-1]) mod M.
    # (Note: the complete sum including A[n-1] gives the steps to return to area 1,
    # and that value equals total mod M – but we do not include it as a distinct rest area.)
    P = [0] * n
    prefix = 0
    P[0] = 0
    for i in range(1, n):
        prefix += A[i - 1]
        P[i] = prefix % M

    # A valid pair (s,t) (with s != t) is counted in one of two cases:
    #
    # 1. When the clockwise walk from s to t involves no wrap-around (s < t).
    #    The number of steps is P[t-1] - P[s-1] (taking proper mod M arithmetic)
    #    and since we want that to be a multiple of M, we require:
    #           (P[t-1] - P[s-1]) mod M == 0  <=> P[t-1] == P[s-1].
    #    Thus, for indices i<j (in our P array) with the same remainder, the pair is valid.
    #
    # 2. When the walk from s to t involves a wrap-around (s > t).
    #    In that case the number of steps is: total - (P[s-1] - P[t-1]).
    #    We need: (total - (P[s-1] - P[t-1])) mod M == 0.
    #    This rearranges (using modulo properties) to:
    #           P[s-1] == P[t-1] + total (mod M).
    #    Since total mod M = T, this is equivalent to:
    #           P[s-1] == (P[t-1] + T) mod M.
    #
    # We now count these pairs.
    
    # (1) Counting pairs (s,t) with s < t and P[s-1]==P[t-1]:
    freq = defaultdict(int)
    for r in P:
        freq[r] += 1
    count_sless = 0
    for r in freq:
        cnt = freq[r]
        count_sless += cnt * (cnt - 1) // 2

    # (2) Counting pairs (s,t) with s > t and P[s-1] == (P[t-1] + T) mod M.
    # We iterate over the indices in increasing order (which means the current index i
    # will represent the candidate "s" and earlier indices represent candidate "t").
    count_sgreater = 0
    cnt_so_far = defaultdict(int)
    for r in P:
        # For current candidate index representing s (with value r),
        # we need to find how many earlier indices (t) have value (r - T) % M.
        needed = (r - T) % M
        count_sgreater += cnt_so_far.get(needed, 0)
        cnt_so_far[r] += 1

    ans = count_sless + count_sgreater
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()