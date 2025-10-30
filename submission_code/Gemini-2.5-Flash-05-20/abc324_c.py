import sys

# Parameters for hashing
# Choose a prime base
BASE = 37
# Choose a large prime modulus
MOD = 10**9 + 7

# Precompute powers of BASE modulo MOD up to MAX_LEN
MAX_LEN = 5 * 10**5 + 5 # Max possible string length from constraints + a small buffer
powers_B = [1] * (MAX_LEN + 1)
for i in range(1, MAX_LEN + 1):
    powers_B[i] = (powers_B[i-1] * BASE) % MOD

def calculate_hashes(s_val):
    """
    Calculates prefix hashes and reversed prefix hashes for a given string.
    `pref_hashes[k]` stores hash of s_val[0...k-1].
    `rev_pref_hashes[k]` stores hash of (s_val[L-1]...s_val[L-1-k]), i.e., prefix of reversed string.
    """
    L = len(s_val)
    pref_hashes = [0] * (L + 1)
    rev_pref_hashes = [0] * (L + 1) # Prefixes of the reversed string
    
    for i in range(L):
        # Hash for s_val[0...i]
        pref_hashes[i+1] = (pref_hashes[i] * BASE + (ord(s_val[i]) - ord('a') + 1)) % MOD
        # Hash for s_val[L-1]...s_val[L-1-i] (prefix of reversed string)
        rev_pref_hashes[i+1] = (rev_pref_hashes[i] * BASE + (ord(s_val[L-1-i]) - ord('a') + 1)) % MOD
    return pref_hashes, rev_pref_hashes

def get_substring_hash(h_pref, start_idx, end_idx):
    """
    Returns hash of string[start_idx : end_idx] (exclusive end_idx) using precomputed prefix hashes.
    """
    length = end_idx - start_idx
    if length <= 0:
        return 0
    val = (h_pref[end_idx] - (h_pref[start_idx] * powers_B[length]) % MOD + MOD) % MOD
    return val

def check_one_edit_distance(s_str, t_prime_str, s_ph, s_rph, t_prime_ph, t_prime_rph):
    """
    Checks if s_str and t_prime_str are at most one edit distance apart using hashing.
    s_ph: prefix hashes for s_str
    s_rph: reversed prefix hashes for s_str
    t_prime_ph: prefix hashes for t_prime_str
    t_prime_rph: reversed prefix hashes for t_prime_str
    """
    ls, lt_prime = len(s_str), len(t_prime_str)

    # Initial length check: if lengths differ by more than 1, it's impossible.
    if abs(ls - lt_prime) > 1:
        return False

    # Helper functions to get hashes of substrings of s_str and t_prime_str
    def get_s_sub_hash(start, end):
        return get_substring_hash(s_ph, start, end)
    def get_t_prime_sub_hash(start, end):
        return get_substring_hash(t_prime_ph, start, end)

    # Helper functions to get hashes of substrings from the end (effectively suffix hashes)
    def get_s_rev_sub_hash(start_orig, end_orig): 
        # The segment s_str[start_orig:end_orig] corresponds to a prefix in the reversed string.
        actual_start_in_rev = ls - end_orig
        actual_end_in_rev = ls - start_orig
        return get_substring_hash(s_rph, actual_start_in_rev, actual_end_in_rev)
    
    def get_t_prime_rev_sub_hash(start_orig, end_orig): 
        actual_start_in_rev = lt_prime - end_orig
        actual_end_in_rev = lt_prime - start_orig
        return get_substring_hash(t_prime_rph, actual_start_in_rev, actual_end_in_rev)

    # Binary search for LCP (Longest Common Prefix)
    low, high = 0, min(ls, lt_prime)
    lcp = 0
    while low <= high:
        mid = (low + high) // 2
        if get_s_sub_hash(0, mid) == get_t_prime_sub_hash(0, mid):
            lcp = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Binary search for LCS (Longest Common Suffix)
    # `mid` here represents the length of the suffix being considered.
    # The suffix comparison starts from the end of the original strings.
    # We limit the search for LCS so that it does not overlap with the LCP.
    low, high = 0, min(ls - lcp, lt_prime - lcp) 
    lcs = 0
    while low <= high:
        mid = (low + high) // 2
        # Compare s_str[ls-mid : ls] with t_prime_str[lt_prime-mid : lt_prime]
        if get_s_rev_sub_hash(ls - mid, ls) == get_t_prime_rev_sub_hash(lt_prime - mid, lt_prime):
            lcs = mid
            low = mid + 1
        else:
            high = mid - 1

    # Condition for one (or zero) edit distance:
    # The total length of matched prefix and suffix must be at least the length of the shorter string.
    return lcp + lcs >= min(ls, lt_prime)


def main():
    # Use sys.stdin.readline for fast input
    input = sys.stdin.readline

    # Read N and T'
    N_str, T_prime_str = input().split()
    N = int(N_str)
    T_prime_str = T_prime_str.strip() # Remove newline characters

    # Precompute hashes for T' once
    T_prime_ph, T_prime_rph = calculate_hashes(T_prime_str)

    results = [] # List to store indices of matching S_i
    for i in range(1, N + 1):
        S_i_str = input().strip() # Read each S_i string
        # Precompute hashes for current S_i
        S_i_ph, S_i_rph = calculate_hashes(S_i_str)

        # Check if S_i could be T
        if check_one_edit_distance(S_i_str, T_prime_str, S_i_ph, S_i_rph, T_prime_ph, T_prime_rph):
            results.append(i)

    # Output the results
    sys.stdout.write(str(len(results)) + "
")
    sys.stdout.write(" ".join(map(str, results)) + "
")

if __name__ == '__main__':
    main()