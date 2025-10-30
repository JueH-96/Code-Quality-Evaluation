import math

class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Compute the original number of '1's in s
        original_ones = s.count('1')
        
        # Create the augmented string t = '1' + s + '1'
        t = '1' + s + '1'
        
        # Find all maximal runs in t
        runs = []
        i = 0
        len_t = len(t)
        while i < len_t:
            char = t[i]
            count = 0
            j = i
            while j < len_t and t[j] == char:
                count += 1
                j += 1
            runs.append((char, count))  # Store (character, size) for each run
            i = j  # Move to the next starting point
        
        # Extract the sizes of '1' runs and '0' runs
        one_runs = [size for c, size in runs if c == '1']
        zero_runs = [size for c, size in runs if c == '0']
        
        m = len(zero_runs)  # Number of '0' runs
        
        # If fewer than 2 '0' runs, no trade is possible
        if m < 2:
            return original_ones
        
        # Compute prefix maximum for zero_runs
        zero_sizes = zero_runs
        prefix_max_zero = [0] * m
        prefix_max_zero[0] = zero_sizes[0]
        for i in range(1, m):
            prefix_max_zero[i] = max(prefix_max_zero[i - 1], zero_sizes[i])
        
        # Compute suffix maximum for zero_runs
        suffix_max_zero = [0] * m
        suffix_max_zero[m - 1] = zero_sizes[m - 1]
        for i in range(m - 2, -1, -1):
            suffix_max_zero[i] = max(zero_sizes[i], suffix_max_zero[i + 1])
        
        # Initialize answer with the no-trade case
        ans = original_ones
        
        # Iterate over each possible trade, p from 0 to m-2
        for p in range(0, m - 1):  # range(0, m-1) gives 0 to m-2 inclusive
            A = zero_sizes[p]  # Size of left adjacent '0' run
            B = zero_sizes[p + 1]  # Size of right adjacent '0' run
            L1 = one_runs[p + 1]  # Size of the '1' run being removed
            new_size = A + L1 + B  # New '0' run size after removal
            
            # Compute the maximum '0' run size excluding the p-th and (p+1)-th '0' runs
            max_excluded = -1  # Use -1 as sentinel since all sizes >= 1
            
            # Max over left part: indices 0 to p-1
            if p - 1 >= 0:
                left_max = prefix_max_zero[p - 1]  # Max from 0 to p-1
                max_excluded = max(max_excluded, left_max)
            
            # Max over right part: indices p+2 to m-1
            if p + 2 <= m - 1:
                right_max = suffix_max_zero[p + 2]  # Max from p+2 to m-1
                max_excluded = max(max_excluded, right_max)
            
            # M_p is the maximum '0' run size after the removal and merging
            M_p = max(max_excluded, new_size)
            
            # Net gain from the trade
            net_gain = M_p - L1
            
            # Final count after this trade
            final_count_p = original_ones + net_gain
            
            # Update the answer with the maximum
            ans = max(ans, final_count_p)
        
        return ans