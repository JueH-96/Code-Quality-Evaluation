class Solution:
  def longestPalindrome(self, s: str, t: str) -> int:
    n = len(s)
    m = len(t)
    max_l = 0

    # is_palindrome_check(s_start_idx, s_end_idx, t_start_idx, t_end_idx)
    # Checks if s[s_start_idx : s_end_idx] + t[t_start_idx : t_end_idx] is a palindrome.
    # Indices for s and t are for the original strings.
    # s_end_idx and t_end_idx are exclusive, as in Python slicing.
    def is_palindrome_check(s_i: int, s_j: int, t_k: int, t_l: int) -> bool:
        len_s_sub = s_j - s_i
        len_t_sub = t_l - t_k
        current_total_len = len_s_sub + len_t_sub

        if current_total_len == 0:
            # An empty string could be considered a palindrome of length 0.
            # However, problem constraints (s,t length >= 1) imply we can always find a palindrome of at least length 1.
            # For example, s[0:1] (first char of s) + "" (empty t_sub) is s[0], which is a palindrome of length 1.
            # So, we can safely ignore 0-length palindromes for updating max_l.
            # Returning False here ensures that 0-length combined strings don't affect max_l.
            return False 
                                 
        # Loop for half the length, comparing characters from ends inwards
        for c_idx in range(current_total_len // 2):
            # Character from the left end of the combined string (index c_idx)
            char1 = ''
            if c_idx < len_s_sub:
                char1 = s[s_i + c_idx]
            else: # c_idx corresponds to an index in t_sub
                # The index in t_sub is (c_idx - len_s_sub), relative to t_k
                char1 = t[t_k + (c_idx - len_s_sub)]
            
            # Character from the right end of the combined string
            # Its index from the left is mirror_c_idx = current_total_len - 1 - c_idx
            mirror_c_idx = current_total_len - 1 - c_idx
            char2 = ''
            if mirror_c_idx < len_s_sub: # mirror_c_idx corresponds to an index in s_sub
                # The index in s_sub is mirror_c_idx, relative to s_i
                char2 = s[s_i + mirror_c_idx]
            else: # mirror_c_idx corresponds to an index in t_sub
                # The index in t_sub is (mirror_c_idx - len_s_sub), relative to t_k
                char2 = t[t_k + (mirror_c_idx - len_s_sub)]
            
            if char1 != char2:
                return False
        return True

    # Iterate over all possible substrings of s (defined by s[i:j])
    # s_sub = s[i:j]
    for i in range(n + 1): # s_sub start index i
        for j in range(i, n + 1): # s_sub end index j (exclusive for s[i:j])
            
            # Iterate over all possible substrings of t (defined by t[k_iter:l_iter])
            # t_sub = t[k_iter:l_iter]
            for k_iter in range(m + 1): # t_sub start index k_iter
                for l_iter in range(k_iter, m + 1): # t_sub end index l_iter (exclusive for t[k_iter:l_iter])
                    
                    # Check if s[i:j] + t[k_iter:l_iter] is a palindrome
                    if is_palindrome_check(i, j, k_iter, l_iter):
                        current_len = (j - i) + (l_iter - k_iter)
                        if current_len > max_l:
                            max_l = current_len
    
    return max_l