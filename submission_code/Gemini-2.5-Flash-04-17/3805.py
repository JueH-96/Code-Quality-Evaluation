import collections

class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        initial_ones = s.count('1')

        t = '1' + s + '1'
        
        # Get block lengths in t
        Ls = []
        if not t:
             # Should not happen based on constraints (n >= 1)
             return 0
        
        current_char = t[0]
        current_len = 0
        for char in t:
            if char == current_char:
                current_len += 1
            else:
                Ls.append(current_len)
                current_char = char
                current_len = 1
        Ls.append(current_len)
        
        k = len(Ls) - 1 # Index of the last block (Ls has k+1 elements)

        # Candidate blocks for Step 1 are Ls[i] where i is an even index
        # corresponding to a '1' block *within* the original string s
        # and surrounded by '0's in s.
        # In t = '1' + s + '1' = 1^{L_0} 0^{L_1} 1^{L_2} 0^{L_3} ... 0^{L_{k-1}} 1^{L_k},
        # '1' blocks in s surrounded by '0's correspond to Ls[i] where i is even
        # and 0 < i < k. This means i = 2, 4, ..., k-2.
        # This range is non-empty only if k-2 >= 2, i.e., k >= 4.
        # If k < 4, there are no such candidate '1' blocks in s.
        if k < 4:
            return initial_ones

        # Extract zero block lengths Ls[1], Ls[3], ..., Ls[k-1]
        # These are the original zero blocks in s (or adjacent to augmented 1s)
        Z_list = [Ls[j] for j in range(1, k, 2)] # indices 1, 3, ..., k-1
        m = len(Z_list) - 1 # index of the last element in Z_list (size is m+1)

        # Calculate prefix maximums of Z_list
        # PrefixMax[j] stores max(Z_list[0...j])
        PrefixMax = [0] * (m + 1)
        if m >= 0: # Z_list is not empty if k >= 2
            PrefixMax[0] = Z_list[0]
            for j in range(1, m + 1):
                PrefixMax[j] = max(PrefixMax[j-1], Z_list[j])

        # Calculate suffix maximums of Z_list
        # SuffixMax[j] stores max(Z_list[j...m])
        SuffixMax = [0] * (m + 1)
        if m >= 0: # Z_list is not empty if k >= 2
            SuffixMax[m] = Z_list[m]
            for j in range(m - 1, -1, -1):
                SuffixMax[j] = max(SuffixMax[j+1], Z_list[j])

        max_gain = 0 # Maximum increase over initial_ones

        # Iterate through candidate Ls[i] blocks for Step 1
        # These are Ls[i] where i is even, 0 < i < k.
        # The index i in Ls corresponds to L_{2j+2} for some j.
        # i = 2 => j = 0
        # i = 4 => j = 1
        # ...
        # i = k-2 => k-2 = 2j+2 => k-4 = 2j => j = (k-4)/2 = m-1
        # j iterates from 0 to m-1
        for j in range(m): 
            i = 2 * j + 2 # Corresponds to index in Ls

            current_Li = Ls[i]
            L_im1 = Ls[i-1] # This is Z_list[j] (Length of zero block before Ls[i])
            L_ip1 = Ls[i+1] # This is Z_list[j+1] (Length of zero block after Ls[i])

            # When Ls[i] is converted to zeros, it merges with Ls[i-1] and Ls[i+1]
            merged_zero_len = L_im1 + current_Li + L_ip1

            # Find max length among other original zero blocks (Z_list excluding Z_list[j] and Z_list[j+1])
            max_other_zeros = 0
            # Max of Z_list[0...j-1]
            if j > 0:
                max_other_zeros = max(max_other_zeros, PrefixMax[j-1])
            # Max of Z_list[j+2...m]
            if j + 2 <= m:
                max_other_zeros = max(max_other_zeros, SuffixMax[j+2])
            
            # The maximum length of a zero block surrounded by ones in the modified string
            Q_max_i = max(merged_zero_len, max_other_zeros)
            
            # The change in the number of ones in s:
            # We lose current_Li ones from s, but gain Q_max_i ones by filling the largest zero gap
            gain_i = Q_max_i - current_Li
            
            # We want the maximum possible increase over the initial count
            max_gain = max(max_gain, gain_i)

        # The maximum number of active sections is the initial count plus the maximum gain achievable
        # If max_gain is 0 or negative, it means no trade improves the count, so we implicitly don't trade
        # by adding max(0, max_gain) to initial count. But max_gain is already max(0, ...) due to initialization.
        return initial_ones + max_gain