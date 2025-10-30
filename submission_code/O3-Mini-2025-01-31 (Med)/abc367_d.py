def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # We have N rest areas and A[i] is the number of steps from area i to area i+1
    # (with area N+1 being area 1). For any ordered pair (s, t) (s ≠ t) the clockwise
    # distance from s to t is defined as:
    #   if s < t: distance = sum_{i=s}^{t-1} A_i
    #   if s > t: distance = TotalSum - sum_{i=t}^{s-1} A_i
    # Our goal is to count those pairs (s, t) for which the clockwise distance is a multiple of M.
    
    # We can compute a prefix sum for a one‐round journey. Let P[0] = 0 (for rest area 1)
    # and for i from 1 to N, P[i] = sum_{j=1}^{i} A_j. Then consider only the first N values
    # (indices 0 to N-1 corresponding to rest areas 1 through N).
    #
    # 1) Pairs that do NOT cross the wrap‐around (i.e. when s < t):
    #    The distance from s to t is P[t] - P[s] and it is a multiple of M if and only if:
    #         P[t] ≡ P[s] (mod M)
    #
    # 2) Pairs that cross the wrap‐around (i.e. when s > t):
    #    The distance is T - (P[s] - P[t]) where T = P[N] is the total perimeter.
    #    We require T - (P[s] - P[t]) ≡ 0 (mod M) which is equivalent to:
    #         (P[s] - P[t]) ≡ T (mod M)
    #    This can be rearranged to: P[s] ≡ T + P[t] (mod M)
    #
    # We will precompute prefix_mod = [P[0] mod M, P[1] mod M, ..., P[N-1] mod M]
    # and count pairs of indices for both cases.
    
    prefix_mod = [0] * N
    cur = 0
    prefix_mod[0] = 0
    for i in range(1, N):
        cur += A[i-1]  # adding distance from rest area i to i+1 (1-indexed)
        prefix_mod[i] = cur % M
        
    # Total perimeter
    total = cur + A[N-1]  # adding the final step from area N to area 1
    T_mod = total % M

    # Case 1: (s < t) -> For indices i, j with i < j, we need prefix_mod[i] == prefix_mod[j]
    counts = {}
    for r in prefix_mod:
        counts[r] = counts.get(r, 0) + 1
    ans_case1 = 0
    for c in counts.values():
        if c > 1:
            ans_case1 += c * (c - 1) // 2
    
    # Case 2: (s > t) -> For indices i, j with i > j, we need:
    #         prefix_mod[i] ≡ T_mod + prefix_mod[j]  (mod M)
    # We process the indices in descending order so that when handling an index j,
    # we have already counted all indices i > j in a frequency dictionary.
    ans_case2 = 0
    freq = {}
    for j in range(N-1, -1, -1):
        need = (prefix_mod[j] + T_mod) % M
        ans_case2 += freq.get(need, 0)
        freq[prefix_mod[j]] = freq.get(prefix_mod[j], 0) + 1

    # Total answer is the sum of both cases.
    ans = ans_case1 + ans_case2
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()