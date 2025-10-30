import math
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # Count substrings with no zeros (all‐ones substrings).
        count_no_zero = 0
        i = 0
        while i < n:
            if s[i] == '1':
                j = i
                while j < n and s[j] == '1':
                    j += 1
                length = j - i
                count_no_zero += length * (length + 1) // 2
                i = j
            else:
                i += 1

        # Build list of indices where s has '0'
        Zpos = [i for i, ch in enumerate(s) if ch == '0']
        L = len(Zpos)
        count_with_zero = 0
        
        # For any substring with at least one 0,
        # write zeros count = k and ones count = m; the condition is:
        #   m >= k^2.
        # But the total length of substring = k + m must be at least k + k^2.
        # Note that if k + k^2 > n (the length of s) then no such substring can exist.
        # Hence we only need to consider k in 1..K_max, with K_max = floor((sqrt(1+4*n)-1)/2)
        if L > 0:
            max_possible_k = int((math.sqrt(1 + 4 * n) - 1) // 2)
            max_k = min(max_possible_k, L)
            # For substrings that contain exactly k zeros,
            # we use the following idea:
            # • Let the zeros in s be at positions Zpos[0], Zpos[1], …, Zpos[L-1].
            # • A substring that contains exactly k zeros must have its first zero coming from one position in Zpos,
            #   its last zero coming from a later one, and no extra zeros outside.
            # For a particular block of exactly k zeros chosen by taking indices i0, i0+1, …, i0+k–1:
            #   • The left boundary x of the substring can be chosen from
            #       [L_bound, B_bound]  where L_bound = (Zpos[i0-1]+1 if i0>0 else 0)
            #             and B_bound = Zpos[i0].
            #   • The right boundary y can be chosen from
            #       [C_val, R_bound] where C_val = Zpos[i0+k-1] and
            #             R_bound = (Zpos[i0+k]-1 if i0+k < L else n-1).
            #
            # However, we only count those substrings whose length (y - x + 1) is at least T = k + k^2.
            #
            # For a given starting index x, the minimal valid y is:
            #    y_min = max(C_val, x + T - 1)
            # and the number of valid y is:
            #    f(x) = max(0, R_bound - max(C_val, x + T - 1) + 1).
            #
            # We want to sum f(x) for x in [A_bound, B_bound].
            #
            # Notice that when x + T - 1 <= C_val then f(x) = R_bound - C_val + 1 (a constant)
            # and when x + T - 1 > C_val then f(x) = R_bound - (x + T - 1) + 1 = R_bound - x - T + 2.
            #
            # We can split the summation on x into two parts.
            
            for k in range(1, max_k + 1):
                T = k + k * k  # minimal required substring length (k zeros + at least k^2 ones)
                # For each block of exactly k zeros
                for i0 in range(0, L - k + 1):
                    # Allowed starting positions x:
                    # They must be chosen from the interval:
                    #   [A_bound, B_bound]  where:
                    #     A_bound = (Zpos[i0-1] + 1) if i0 > 0, else 0.
                    #     B_bound = Zpos[i0]
                    A_bound = Zpos[i0 - 1] + 1 if i0 > 0 else 0
                    B_bound = Zpos[i0]
                    
                    # The substring must include zeros at positions i0,..., i0+k-1.
                    # So the earliest possible y is C_val, and the latest allowed is R_bound:
                    #    C_val = Zpos[i0+k-1]
                    #    R_bound = (Zpos[i0+k]-1) if (i0+k < L) else (n-1)
                    C_val = Zpos[i0 + k - 1]
                    R_bound = Zpos[i0 + k] - 1 if i0 + k < L else n - 1
                    
                    if C_val > R_bound:
                        continue  # no valid ending indices
                    
                    block_sum = 0
                    # For each starting index x in [A_bound, B_bound], we need y in [max(C_val, x + T - 1), R_bound].
                    # Write f(x) = max(0, R_bound - max(C_val, x + T - 1) + 1)
                    #
                    # Split x:
                    # 1. x such that x + T - 1 <= C_val (i.e. x <= C_val - T + 1).
                    #    Then f(x) is constant: R_bound - C_val + 1.
                    x1_end = min(B_bound, C_val - T + 1)
                    if x1_end >= A_bound:
                        const_val = R_bound - C_val + 1
                        block_sum += (x1_end - A_bound + 1) * const_val
                    
                    # 2. x such that x + T - 1 > C_val.
                    #    For these, f(x) = R_bound - (x + T - 1) + 1 = R_bound - x - T + 2.
                    # Also, we only get a positive count when x <= R_bound - T + 1.
                    x2_start = max(A_bound, C_val - T + 2)
                    x2_end = min(B_bound, R_bound - T + 1)
                    if x2_end >= x2_start:
                        n_terms = x2_end - x2_start + 1
                        # Sum of values: for each x, add [R_bound - T + 2 - x]
                        # This is an arithmetic series whose total is:
                        #   n_terms*(R_bound - T + 2) - (sum of x from x2_start to x2_end)
                        # The sum of x from L to U is (L + U)*n_terms//2.
                        block_sum += n_terms * (R_bound - T + 2) - (x2_start + x2_end) * n_terms // 2
                    
                    count_with_zero += block_sum

        return count_no_zero + count_with_zero