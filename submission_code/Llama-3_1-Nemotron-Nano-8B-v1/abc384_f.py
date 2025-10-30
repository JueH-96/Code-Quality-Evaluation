def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    max_k = 30  # Since 2^30 is larger than 2*1e7
    T = [0] * (max_k + 2)  # T[k] for 0 <= k <= max_k+1
    
    for k in range(max_k + 1):
        mod = 1 << (k + 1)
        residues = [a % mod for a in A]
        cnt = {}
        for r in residues:
            cnt[r] = cnt.get(r, 0) + 1
        
        sum_rs = 0
        # Precompute all residues to avoid duplicate processing
        processed = set()
        for r in cnt:
            s = ((1 << k) - (r % (1 << k))) % (1 << k)
            if r == s and r not in processed:
                sum_rs += r * cnt[r] * (cnt[r] + 1)
                processed.add(r)
            elif r < s and (r, s) not in processed and (s, r) not in processed:
                sum_rs += (1 << k) * cnt[r] * cnt.get(s, 0)
                processed.add((r, s))
                processed.add((s, r))
        
        sum_m = 0
        for a in A:
            r_i = a % mod
            m_i = a // mod
            s_i = ((1 << k) - (r_i % (1 << k))) % (1 << k)
            sum_m += m_i * cnt.get(s_i, 0)
        
        T[k] = sum_rs + (1 << (k + 1)) * sum_m
    
    ans = 0
    for k in range(max_k + 1):
        if k + 1 > max_k:
            T_k_plus_1 = 0
        else:
            T_k_plus_1 = T[k + 1]
        S_k = T[k] - T_k_plus_1
        ans += S_k // (1 << k)
    
    print(ans)

if __name__ == '__main__':
    main()