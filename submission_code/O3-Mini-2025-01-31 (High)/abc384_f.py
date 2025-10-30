# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # read the array A of length n
    A = [int(x) for x in data[1:1+n]]
    
    # Diagonal part: sum_{i=1}^n f(2*A[i]).
    # Note: if a & -a returns the lowest power of two dividing a,
    # then f(2*a) = (2*a) // ((a & -a) << 1) = a // (a & -a).
    diag_sum = 0
    for a in A:
        diag_sum += a // (a & -a)
        
    # Off-diagonals: We wish to compute S_off = sum_{i<j} f(A[i]+A[j]).
    # For a pair (a,b), write a+b = 2^k * m (m odd). Then f(a+b) = (a+b) // 2^k.
    # Moreover, a+b ≡ 2^k (mod 2^(k+1)). Thus each pair (a,b) belongs to a unique
    # k for which (a+b) mod (2^(k+1)) == 2^k.
    # So for each k we compute raw_sum(k) = sum_{i<j, (a+b) mod 2^(k+1)== 2^k} (a+b)
    # and add raw_sum(k) // (2^k) to our answer.
    
    pairs_sum = 0
    if A:
        maxA = max(A)
    else:
        maxA = 0
    # Maximum possible pair sum is at most 2*maxA so we loop for
    # k in 0,1,...,K-1 where K = (2*maxA).bit_length()
    K = (maxA * 2).bit_length()
    
    # For each k, let M = 2^(k+1) (we use mod arithmetic because M is a power‐of‐2)
    for k in range(K):
        mod_val = 1 << (k+1)
        mod_mask = mod_val - 1
        target = 1 << k
        # Build dictionary grouping: for residue r, we store (count, sum of elements)
        d = {}
        for a in A:
            r = a & mod_mask  # a mod mod_val
            prev = d.get(r)
            if prev is None:
                d[r] = (1, a)
            else:
                d[r] = (prev[0] + 1, prev[1] + a)
                
        raw_sum = 0
        # Now loop over each group. For each residue r, let partner t = (target - r) mod mod_val.
        # To count each unordered pair only once, we only add when r < t or when r==t.
        for r, (cnt_r, sum_r) in d.items():
            t = (target - r) & mod_mask
            if r < t:
                partner = d.get(t)
                if partner is not None:
                    cnt_t, sum_t = partner
                    raw_sum += cnt_r * sum_t + cnt_t * sum_r
            elif r == t:
                if cnt_r > 1:
                    raw_sum += (cnt_r - 1) * sum_r
        # Every pair counted here contributes f(a+b) = (a+b) // 2^k.
        pairs_sum += raw_sum >> k  # same as raw_sum // (1 << k)
    
    ans = diag_sum + pairs_sum
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()