def main():
    import sys
    from collections import Counter
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    # Separate predetermined values (non-negative numbers, not -1) from missing entries.
    knownA = [a for a in A if a != -1]
    knownB = [b for b in B if b != -1]
    m = len(knownA)
    k = len(knownB)
    
    # If you can avoid pairing any predetermined A with a predetermined B, then it's always possible.
    # You can do that if m ≤ N–k and k ≤ N–m, equivalently m+k ≤ N.
    if m + k <= N:
        sys.stdout.write("Yes")
        return
    
    # Otherwise, forced pairs of predetermined values occur.
    # The number F = m+k-N is the minimum number of pairs that force A_i+B_i = S.
    F = m + k - N
    
    # Build frequency dictionaries for predetermined A and B.
    freqA = Counter(knownA)
    freqB = Counter(knownB)
    
    # For any forced pair (with predetermined A value x and predetermined B value y) we need
    # x+y = S. In forced pairs these sums are fixed so they force a unique global S.
    # Also, when pairing a predetermined with a missing value, you can choose the missing value
    # to be S – (the predetermined one); so to have nonnegative values we must have S >= max(a, b).
    # Let M_val be that lower bound.
    M_val = 0
    if knownA:
        M_val = max(M_val, max(knownA))
    if knownB:
        M_val = max(M_val, max(knownB))
        
    # For a given candidate S (which necessarily must equal x+y for some x in A, y in B)
    # the maximum forced pairs you can form is:
    #    forced(S) = Σ₍ₓ₎ [if (S-x) is in B then min(freqA[x], freqB[S-x]) else 0]
    # We compute this for each candidate S that comes from some pair (x,y) with x in A and y in B.
    candidate_forced = {}
    freqB_items = list(freqB.items())
    for x, cnt_x in freqA.items():
        for y, cnt_y in freqB_items:
            S_val = x + y
            if S_val < M_val:
                continue
            # Contribution from predetermined A value x with a matching predetermined B value y is
            # min(cnt_x, cnt_y). (For a given x, note that only y = S_val – x qualifies.)
            candidate_forced[S_val] = candidate_forced.get(S_val, 0) + (cnt_x if cnt_x < cnt_y else cnt_y)
    
    # If there exists any candidate S (with S ≥ M_val) for which the maximum forced pairs we can form is at least F,
    # then we can set missing values accordingly so that for every index A_i+B_i = S.
    possible = any(forced_count >= F for s_val, forced_count in candidate_forced.items() if s_val >= M_val)
    sys.stdout.write("Yes" if possible else "No")
    
if __name__ == '__main__':
    main()