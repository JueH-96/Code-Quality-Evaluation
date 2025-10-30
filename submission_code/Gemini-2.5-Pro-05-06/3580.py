class Solution:
  def minStartingIndex(self, s: str, pattern: str) -> int:
    n = len(s)
    m = len(pattern)

    # Constraints: 1 <= pattern.length < s.length <= 10^5
    # So m >= 1, n > m.

    # Using two hash functions to reduce probability of collision.
    BASE1, MOD1 = 31, 10**9 + 7
    BASE2, MOD2 = 37, 10**9 + 9 # Common choices for bases and moduli

    # Precompute powers of BASEs
    # p_powers arrays need to store powers up to BASE^n, as substrings of s can be up to length n.
    # p_powers[k] = BASE^k. Max index needed is n for length n. So, size n+1.
    p_powers1 = [1] * (n + 1)
    p_powers2 = [1] * (n + 1)
    for i in range(1, n + 1):
        p_powers1[i] = (p_powers1[i-1] * BASE1) % MOD1
        p_powers2[i] = (p_powers2[i-1] * BASE2) % MOD2

    # Precompute prefix hashes for s
    # h_s[k] stores hash of s[0...k-1] (prefix of length k)
    # h_s[0] = 0 (hash of empty string)
    # Size n+1 for indices 0 to n.
    h_s1 = [0] * (n + 1)
    h_s2 = [0] * (n + 1)
    for i in range(n):
        char_val = ord(s[i]) - ord('a') + 1
        h_s1[i+1] = (h_s1[i] * BASE1 + char_val) % MOD1
        h_s2[i+1] = (h_s2[i] * BASE2 + char_val) % MOD2

    # Precompute prefix hashes for pattern
    # h_p[k] stores hash of pattern[0...k-1]
    # Size m+1 for indices 0 to m.
    h_p1 = [0] * (m + 1)
    h_p2 = [0] * (m + 1)
    for i in range(m):
        char_val = ord(pattern[i]) - ord('a') + 1
        h_p1[i+1] = (h_p1[i] * BASE1 + char_val) % MOD1
        h_p2[i+1] = (h_p2[i] * BASE2 + char_val) % MOD2

    # Helper to get hash of s[L_idx...R_idx] (0-indexed, inclusive)
    def get_s_hash(L_idx, R_idx): 
        if L_idx > R_idx: # Empty substring
            return (0, 0)
        
        length = R_idx - L_idx + 1
        
        # hash(s[L..R]) = (h_s[R+1] - h_s[L] * P^length) % MOD
        # h_s[k] is hash of prefix of length k (s[0...k-1])
        # So, s[L..R] is s[0..R] minus s[0..L-1].
        # hash_s[R_idx+1] is hash of s[0...R_idx]
        # hash_s[L_idx] is hash of s[0...L_idx-1]
        
        hash1 = (h_s1[R_idx+1] - h_s1[L_idx] * p_powers1[length]) % MOD1
        hash1 = (hash1 + MOD1) % MOD1 # Ensure positive result
        
        hash2 = (h_s2[R_idx+1] - h_s2[L_idx] * p_powers2[length]) % MOD2
        hash2 = (hash2 + MOD2) % MOD2 # Ensure positive result
        
        return (hash1, hash2)

    # Helper to get hash of pattern[L_idx...R_idx] (0-indexed, inclusive)
    def get_p_hash(L_idx, R_idx): 
        if L_idx > R_idx: # Empty substring
            return (0, 0)
            
        length = R_idx - L_idx + 1
        hash1 = (h_p1[R_idx+1] - h_p1[L_idx] * p_powers1[length]) % MOD1
        hash1 = (hash1 + MOD1) % MOD1
        
        hash2 = (h_p2[R_idx+1] - h_p2[L_idx] * p_powers2[length]) % MOD2
        hash2 = (hash2 + MOD2) % MOD2
        
        return (hash1, hash2)

    # Helper to compare s[s_start_idx : s_start_idx+length]
    # with pattern[p_start_idx : p_start_idx+length]
    def check_equal(s_start_idx, p_start_idx, length):
        if length == 0: # Empty substrings are always equal
            return True
        
        # Hashes are for substrings s[s_start_idx ... s_start_idx + length - 1]
        # and pattern[p_start_idx ... p_start_idx + length - 1]
        s_hash_val = get_s_hash(s_start_idx, s_start_idx + length - 1)
        p_hash_val = get_p_hash(p_start_idx, p_start_idx + length - 1)
        
        return s_hash_val == p_hash_val
    
    # Helper to find LCP of s[s_start_idx...] and pattern[p_start_idx...] 
    # up to max_possible_lcp. Returns the length of the LCP.
    def get_lcp_length(s_start_idx, p_start_idx, max_possible_lcp):
        low, high = 0, max_possible_lcp
        lcp_val = 0 # Stores the length of confirmed LCP
        while low <= high:
            mid_len = low + (high - low) // 2
            if mid_len == 0: # An empty prefix (length 0) always matches.
                # lcp_val will be 0 if it's not updated by a positive mid_len.
                low = mid_len + 1 # Try for a longer LCP
                continue

            if check_equal(s_start_idx, p_start_idx, mid_len):
                lcp_val = mid_len # mid_len is a possible LCP length
                low = mid_len + 1 # Try for a longer LCP
            else:
                high = mid_len - 1 # LCP must be shorter than mid_len
        return lcp_val

    # Iterate through all possible starting indices i for substrings of s
    # A substring s[i : i+m] starts at index i and has length m.
    # i ranges from 0 to n-m.
    for i in range(n - m + 1):
        # Current window in s is s[i : i+m]
        # Compare with pattern (length m, indices 0 to m-1)

        # Find the length of the LCP of s[i : i+m] and pattern[0 : m]
        lcp1 = get_lcp_length(i, 0, m)

        if lcp1 == m:
            # The substring s[i:i+m] is identical to pattern. 0 changes needed.
            # This satisfies "at most one change".
            return i
        
        # If lcp1 < m, it means there is a mismatch at index lcp1 (0-indexed relative to window start).
        # s[i+lcp1] != pattern[lcp1]. This is the single allowed change.
        # We need to check if the rest of the string matches:
        # s[i+lcp1+1 : i+m] must match pattern[lcp1+1 : m]
        
        # Suffix in s starts at index: i + lcp1 + 1
        # Suffix in pattern starts at index: lcp1 + 1
        # Length of this suffix part: m - (lcp1 + 1)
        
        s_suffix_query_start_idx = i + lcp1 + 1
        p_suffix_query_start_idx = lcp1 + 1
        suffix_query_len = m - (lcp1 + 1) # length of the part after the mismatch

        if check_equal(s_suffix_query_start_idx, p_suffix_query_start_idx, suffix_query_len):
            # The suffixes match. Combined with LCP, this means exactly one mismatch.
            return i
            
    # No such starting index found
    return -1