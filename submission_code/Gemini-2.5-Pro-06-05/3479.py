import math

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        zero_indices = [i for i, char in enumerate(s) if char == '0']
        num_zeros_total = len(zero_indices)
        
        # A substring is valid if ones >= zeros^2.
        # Let number of zeros be z, ones be o. o >= z*z.
        # Total length L = o+z. o = L-z.
        # L-z >= z^2 => L >= z*z+z.
        # Since L <= n, we have n >= z*z+z, which implies z is roughly at most sqrt(n).
        max_zeros = int(math.sqrt(n))
        
        ans = 0
        
        # `p` is the number of zeros encountered before the current index `i`.
        # It's also the index into `zero_indices` for the first zero at or after `i`.
        p = 0
        for i in range(n):
            # Part 1: Count substrings starting at `i` with 0 zeros.
            # The range of valid ending indices `j` is from `i` to `end_j_n0_0`.
            end_j_n0_0 = n - 1
            if p < num_zeros_total:
                end_j_n0_0 = zero_indices[p] - 1
            
            # For 0 zeros, the condition `ones >= 0^2` is always true.
            # All substrings s[i..j] in this range are valid.
            # The number of choices for j is `end_j_n0_0 - i + 1`.
            ans += max(0, end_j_n0_0 - i + 1)
            
            # Part 2: Count substrings starting at `i` with `n0 > 0` zeros.
            # We only need to check for `n0` up to `max_zeros`.
            for n0 in range(1, max_zeros + 1):
                # The substring must contain the zeros from index `p` to `p + n0 - 1`
                # in the `zero_indices` list.
                
                # Index in `zero_indices` of the last zero for this substring
                last_zero_p_idx = p + n0 - 1
                if last_zero_p_idx >= num_zeros_total:
                    # Not enough zeros left in the string to form a substring with n0 zeros.
                    break
                    
                # The substring's end `j` must be at or after the last required zero.
                j_start = zero_indices[last_zero_p_idx]
                
                # The substring's end `j` must be before the next zero.
                next_zero_p_idx = p + n0
                j_end = n - 1
                if next_zero_p_idx < num_zeros_total:
                    j_end = zero_indices[next_zero_p_idx] - 1
                    
                # We need to satisfy the dominant ones condition: `ones >= n0*n0`.
                # `ones = (j - i + 1) - n0`
                # `(j - i + 1) - n0 >= n0*n0`
                # `j >= n0*n0 + n0 + i - 1`
                
                min_j_req = n0 * n0 + n0 + i - 1
                
                # We need `j` to be in both [j_start, j_end] and [min_j_req, infinity].
                # So, `j` must be in [max(j_start, min_j_req), j_end].
                actual_j_start = max(j_start, min_j_req)
                
                if actual_j_start <= j_end:
                    ans += j_end - actual_j_start + 1

            if s[i] == '0':
                p += 1
                
        return ans