MOD = 10**9 + 7
BASE = 131

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    for i in range(1, m):
        j = lps[i - 1]
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        else:
            j = 0
        lps[i] = j
    return lps

def main():
    import sys
    s = sys.stdin.readline().strip()
    n = len(s)
    if n == 0:
        print("")
        return
        
    rev_s = s[::-1]
    combined = s + '#' + rev_s
    lps_arr = compute_lps(combined)
    match_len = lps_arr[-1]
    
    H_s = [0] * (n + 1)
    pow_base = [1] * (n + 1)
    for i in range(n):
        H_s[i + 1] = (H_s[i] * BASE + ord(s[i])) % MOD
        pow_base[i + 1] = (pow_base[i] * BASE) % MOD
        
    H_rev = [0] * (n + 1)
    for i in range(n):
        H_rev[i + 1] = (H_rev[i] * BASE + ord(rev_s[i])) % MOD
        
    def get_hash(H, L, R):
        return (H[R + 1] - (H[L] * pow_base[R - L + 1]) % MOD) % MOD
        
    ans = None
    for k in range(match_len, -1, -1):
        L_ptr = n - k
        R_ptr = n - 1
        if k == 0:
            is_palindrome = True
        else:
            hash_orig = get_hash(H_s, L_ptr, R_ptr)
            rev_L = n - 1 - R_ptr
            rev_R = n - 1 - L_ptr
            hash_rev_val = get_hash(H_rev, rev_L, rev_R)
            is_palindrome = (hash_orig == hash_rev_val)
            
        if is_palindrome:
            to_append = s[:n - k][::-1]
            ans = s + to_append
            break
            
    if ans is None:
        ans = s + rev_s
    print(ans)

if __name__ == '__main__':
    main()