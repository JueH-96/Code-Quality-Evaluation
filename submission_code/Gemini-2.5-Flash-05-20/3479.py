class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0

        # Precompute indices of zeros for efficient lookup
        # zeros_indices[0] is -1 (virtual zero before string start)
        # zeros_indices[k] is the index of the (k-1)-th zero in s
        # zeros_indices[num_zeros + 1] is n (virtual zero after string end)
        zeros_indices = [-1] + [i for i, char in enumerate(s) if char == '0'] + [n]
        num_zeros_total = len(zeros_indices) - 2 # Actual count of zeros in s

        # Helper function to count (X, Y) pairs where:
        # 0 <= X <= P
        # 0 <= Y <= Q
        # X + Y >= S
        # This function runs in O(1) time.
        def count_pairs_ge(P: int, Q: int, S: int) -> int:
            if S <= 0: # If required sum is 0 or less, all combinations satisfy X+Y >= S
                return (P + 1) * (Q + 1)
            
            # If S is greater than the maximum possible sum (P+Q), no combinations satisfy
            if S > P + Q:
                return 0
            
            total_combinations = (P + 1) * (Q + 1)
            
            # Calculate combinations where X + Y < S (i.e., X + Y <= S - 1)
            # Let S_prime = S - 1
            S_prime = S - 1
            
            count_less_than_S = 0
            
            # We sum `max(0, min(Q, S_prime - X) + 1)` for X from 0 to P
            # The inner term `min(Q, S_prime - X) + 1` splits into two cases based on X:
            
            # Case 1: S_prime - X >= Q (i.e., X <= S_prime - Q)
            # Here, min(Q, S_prime - X) = Q. The term is (Q + 1).
            # This applies for X from 0 to min(P, S_prime - Q).
            x_limit1 = S_prime - Q
            num_terms1 = min(P, x_limit1) + 1
            if num_terms1 > 0:
                count_less_than_S += num_terms1 * (Q + 1)
            
            # Case 2: S_prime - X < Q (i.e., X > S_prime - Q)
            # Here, min(Q, S_prime - X) = S_prime - X. The term is (S_prime - X + 1).
            # This applies for X from max(0, S_prime - Q + 1) to P.
            start_x2 = max(0, S_prime - Q + 1)
            end_x2 = P
            
            if start_x2 <= end_x2:
                # Sum `max(0, S_prime - X + 1)` for X in [start_x2, end_x2].
                # The terms `S_prime - X + 1` decrease as X increases.
                # We only sum positive terms (or 0 if S_prime - X + 1 <= 0).
                
                # The largest X for which S_prime - X + 1 is positive is S_prime.
                effective_end_x2 = min(end_x2, S_prime) 
                
                if start_x2 <= effective_end_x2:
                    # Sum of an arithmetic series: (count * (first_term + last_term)) / 2
                    num_terms2 = effective_end_x2 - start_x2 + 1
                    first_term2 = S_prime - start_x2 + 1
                    last_term2 = S_prime - effective_end_x2 + 1
                    count_less_than_S += (num_terms2 * (first_term2 + last_term2)) // 2
                    
            return total_combinations - count_less_than_S

        # Case 0: Substrings with 0 zeros (k=0)
        # Condition: c1 >= 0^2 (c1 >= 0), which is always true.
        # These are contiguous blocks of '1's.
        current_streak = 0
        for char_s in s:
            if char_s == '1':
                current_streak += 1
            else:
                ans += current_streak * (current_streak + 1) // 2
                current_streak = 0
        ans += current_streak * (current_streak + 1) // 2 # Add for any trailing streak

        # Case 1: Substrings with k zeros, where 1 <= k <= sqrt(N)
        # MAX_K is the maximum number of zeros we need to consider in a substring
        MAX_K = int(n**0.5)

        for k in range(1, MAX_K + 1):
            target_ones = k * k # Minimum number of ones required

            # Iterate through all possible windows of `k` zeros
            # `idx_start_in_zeros_indices` is the 1-based index in `zeros_indices`
            # corresponding to the first zero in the current `k`-zero group.
            # It ranges from 1 up to `num_zeros_total - k + 1`.
            for idx_start_in_zeros_indices in range(1, num_zeros_total - k + 2):
                idx_end_in_zeros_indices = idx_start_in_zeros_indices + k - 1
                
                # `left_zero_pos` is the actual index of the first '0' in this k-group.
                # `right_zero_pos` is the actual index of the last '0' in this k-group.
                left_zero_pos = zeros_indices[idx_start_in_zeros_indices]
                right_zero_pos = zeros_indices[idx_end_in_zeros_indices]
                
                # `c1_at_core` is the count of ones within the segment s[left_zero_pos .. right_zero_pos]
                # Length of segment = (right_zero_pos - left_zero_pos + 1)
                # Ones = Length - Zeros_count = Length - k
                c1_at_core = (right_zero_pos - left_zero_pos + 1) - k
                
                # P_val: Max number of ones that can be added to the left of `left_zero_pos`.
                # This is the length of the string of '1's immediately preceding `s[left_zero_pos]`.
                # It's (left_zero_pos - 1) - (index of zero before it)
                P_val = left_zero_pos - (zeros_indices[idx_start_in_zeros_indices - 1] + 1)
                
                # Q_val: Max number of ones that can be added to the right of `right_zero_pos`.
                # This is the length of the string of '1's immediately succeeding `s[right_zero_pos]`.
                # It's (index of zero after it - 1) - right_zero_pos
                Q_val = (zeros_indices[idx_end_in_zeros_indices + 1] - 1) - right_zero_pos
                
                # The total number of ones in the final substring will be c1_at_core + X + Y,
                # where X is ones taken from left, Y is ones taken from right.
                # We need: c1_at_core + X + Y >= target_ones
                # Which simplifies to: X + Y >= target_ones - c1_at_core
                required_sum_ones = target_ones - c1_at_core
                
                # Add count of (X, Y) pairs that satisfy the condition
                ans += count_pairs_ge(P_val, Q_val, required_sum_ones)
        
        return ans