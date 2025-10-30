class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        
        # Count initial number of 1s
        initial_ones = s.count('1')
        
        # Augment the string
        augmented = '1' + s + '1'
        
        # Find all blocks of 1s surrounded by 0s (candidates to remove)
        ones_blocks = []
        i = 1  # Start from 1 to skip the augmented '1'
        while i <= n:
            if augmented[i] == '1' and augmented[i-1] == '0':
                # Found start of a 1s block
                start = i
                while i <= n and augmented[i] == '1':
                    i += 1
                if i <= n + 1 and augmented[i] == '0':
                    # This block is surrounded by 0s
                    ones_blocks.append(i - start)
            else:
                i += 1
        
        # Find all blocks of 0s surrounded by 1s (candidates to add)
        zeros_blocks = []
        i = 1  # Start from 1 to skip the augmented '1'
        while i <= n:
            if augmented[i] == '0' and augmented[i-1] == '1':
                # Found start of a 0s block
                start = i
                while i <= n and augmented[i] == '0':
                    i += 1
                if i <= n + 1 and augmented[i] == '1':
                    # This block is surrounded by 1s
                    zeros_blocks.append(i - start)
            else:
                i += 1
        
        # If no valid trade is possible, return initial count
        if not ones_blocks or not zeros_blocks:
            return initial_ones
        
        # Find the best trade: remove smallest 1s block, add largest 0s block
        min_ones_to_remove = min(ones_blocks)
        max_zeros_to_add = max(zeros_blocks)
        
        # Calculate the result after the best trade
        result = initial_ones - min_ones_to_remove + max_zeros_to_add
        
        # Also consider not making any trade
        return max(result, initial_ones)