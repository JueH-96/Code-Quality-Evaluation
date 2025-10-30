def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    B = [int(next(it)) for _ in range(n)]
    mod = 998244353

    # Precompute prefix info (segment positions 1..t).
    # prefix_free[t] = number of free positions (B[i]==-1) in B[0..t-1]
    # prefix_fixed_min[t] = minimum fixed value in B[0..t-1] (or None if there is none)
    prefix_free = [0]*(n+1)
    prefix_fixed_min = [None]*(n+1)
    for i in range(n):
        prefix_free[i+1] = prefix_free[i] + (1 if B[i] == -1 else 0)
        if B[i] != -1:
            if prefix_fixed_min[i] is None:
                prefix_fixed_min[i+1] = B[i]
            else:
                cur = prefix_fixed_min[i]
                prefix_fixed_min[i+1] = B[i] if B[i] < cur else cur
        else:
            prefix_fixed_min[i+1] = prefix_fixed_min[i]

    # Precompute suffix info (segment positions t..n).
    # suffix_free[i] = number of free positions in B[i-1..n-1]
    # suffix_fixed_max[i] = maximum fixed value in B[i-1..n-1] (or None if none)
    suffix_free = [0]*(n+2)
    suffix_fixed_max = [None]*(n+2)
    # Process from the end.
    suffix_free[n] = 0
    suffix_fixed_max[n] = None
    for i in range(n-1, -1, -1):
        # i+1 in our 1-indexed scheme.
        suffix_free[i+1] = suffix_free[i+2] + (1 if B[i] == -1 else 0) if i < n-1 else (1 if B[i]== -1 else 0)
        if B[i] != -1:
            if suffix_fixed_max[i+2] is None:
                suffix_fixed_max[i+1] = B[i]
            else:
                cand = suffix_fixed_max[i+2]
                suffix_fixed_max[i+1] = B[i] if B[i] > cand else cand
        else:
            suffix_fixed_max[i+1] = suffix_fixed_max[i+2]

    # Total number of completions for the full sequence.
    total_free = prefix_free[n]
    T_total = pow(M, total_free, mod)
    
    # Precompute powers.
    # For any free count r we need x^r modulo mod for x in 0..M+1.
    # The maximum free count needed is max(prefix_free, suffix_free) <= total_free.
    max_r = total_free  
    power_table = []
    # r==0: define  x^0 = 1 for all x.
    power_table.append([1]*(M+2))
    for r in range(1, max_r+1):
        row = [0]*(M+2)
        for x in range(M+2):
            row[x] = pow(x, r, mod)
        power_table.append(row)
    
    # Now, sum over possible split points.
    # For each t (1 <= t < n), let L = B[0..t-1] and R = B[t..n-1].
    # For L, if there is a fixed value, then its min is forced.
    # Otherwise: Fmin(a) = ( (M-a+1)^(L_free) - (M-a)^(L_free) ).
    # Similarly for R: if fixed then Fmax(b)= b^(R_free) (only for b = forced max),
    # else Fmax(b)= (b^(R_free) - (b-1)^(R_free) ).
    sum_splits = 0
    for t in range(1, n):
        L_free = prefix_free[t]      # free count in left segment
        R_free = suffix_free[t+1]    # free count in right segment (B[t..n-1])
        L_fixed = prefix_fixed_min[t]
        R_fixed = suffix_fixed_max[t+1]
        # Build left_count for a = 1..M: number of completions in the left segment with minimum exactly a.
        left_count = [0]*(M+1)  # index 0 unused
        if L_fixed is not None:
            a_val = L_fixed
            # free positions can choose any value in [a_val, M]
            left_count[a_val] = power_table[L_free][M - a_val + 1]
        else:
            for a in range(1, M+1):
                # Assignments with all free positions in [a, M] minus assignments in [a+1, M]
                val = power_table[L_free][M - a + 1] - power_table[L_free][M - a]
                left_count[a] = val % mod

        # Build right_count for b = 1..M: number of completions in the right segment with maximum exactly b.
        right_count = [0]*(M+1)
        if R_fixed is not None:
            b_val = R_fixed
            right_count[b_val] = power_table[R_free][b_val]
        else:
            for b in range(1, M+1):
                val = power_table[R_free][b] - power_table[R_free][b - 1]
                right_count[b] = val % mod

        # Compute prefix sums for right_count.
        RS = [0]*(M+1)
        s_val = 0
        for x in range(1, M+1):
            s_val = (s_val + right_count[x]) % mod
            RS[x] = s_val

        # For each possible left minimum a we only allow right maximum b < a.
        local_sum = 0
        for a in range(1, M+1):
            if left_count[a]:
                local_sum = (local_sum + left_count[a] * RS[a-1]) % mod
        sum_splits = (sum_splits + local_sum) % mod

    ans = (T_total + sum_splits) % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()