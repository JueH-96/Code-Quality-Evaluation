def main():
    import sys
    from math import gcd
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    # Read all numbers
    A = [int(next(it)) for _ in range(n)]
    
    # Special-case optimizations:
    # If k == 1 then the only subset containing A[i] is {A[i]} so the maximum gcd is A[i] itself.
    if k == 1:
        sys.stdout.write("
".join(str(a) for a in A))
        return
    # If k == n then the only subset is the full array so all answers equal gcd(A)
    if k == n:
        g = A[0]
        for a in A[1:]:
            g = gcd(g, a)
        sys.stdout.write("
".join(str(g) for _ in range(n)))
        return

    # Otherwise k is between 2 and n-1.
    # Let M be the maximum number in A.
    M = max(A)
    
    # Build a frequency array for values in [0, M]
    freq = [0] * (M + 1)
    for a in A:
        freq[a] += 1

    # For every d (1 <= d <= M), we want to know how many elements in A are divisible by d.
    # We do a "sieve‐style" loop: for d from 1 to M, add freq[m] for every multiple m of d.
    cnt = [0] * (M + 1)
    M_val = M  # local alias for speed
    f = freq   # local alias
    for d in range(1, M_val + 1):
        s = 0
        # Loop over multiples of d: d, 2*d, 3*d, ... up to M.
        for m in range(d, M_val + 1, d):
            s += f[m]
        cnt[d] = s

    # Now, for each possible number x (1 <= x <= M) we want to know:
    # “What is the maximum d (which must be a divisor of x) such that at least k numbers in A are divisible by d?”
    # The idea: if A[i] = x, then any viable d must divide x.
    # We precompute best[x] = maximum divisor d of x for which cnt[d] >= k.
    best = [0] * (M + 1)
    bf = best  # local alias for speed
    for d in range(M_val, 0, -1):  # descending order: the first time we set best[m], it is the maximum possible d.
        if cnt[d] >= k:
            step = d
            for m in range(d, M_val + 1, step):
                if bf[m] == 0:
                    bf[m] = d

    # For every A[i], answer is best[A[i]]
    out_lines = []
    for a in A:
        out_lines.append(str(bf[a]))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()