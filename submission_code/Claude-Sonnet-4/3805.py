class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        original_count = s.count('1')
        
        # Augment the string
        augmented = '1' + s + '1'
        
        max_gain = 0
        
        # Find all blocks of '1's surrounded by '0's
        i = 1  # Start from index 1 (skip the first augmented '1')
        while i < len(augmented) - 1:  # End before the last augmented '1'
            if augmented[i] == '1':
                # Find the end of this block of '1's
                j = i
                while j < len(augmented) - 1 and augmented[j] == '1':
                    j += 1
                
                # Check if this block is surrounded by '0's
                if augmented[i-1] == '0' and augmented[j] == '0':
                    # This is a valid block to convert to '0's
                    block_size = j - i
                    
                    # Create the string after converting this block to '0's
                    temp_augmented = augmented[:i] + '0' * block_size + augmented[j:]
                    
                    # Find all blocks of '0's surrounded by '1's in temp_augmented
                    k = 1
                    while k < len(temp_augmented) - 1:
                        if temp_augmented[k] == '0':
                            # Find the end of this block of '0's
                            l = k
                            while l < len(temp_augmented) - 1 and temp_augmented[l] == '0':
                                l += 1
                            
                            # Check if this block is surrounded by '1's
                            if temp_augmented[k-1] == '1' and temp_augmented[l] == '1':
                                # Calculate gain for this trade
                                zeros_converted = l - k
                                ones_converted = block_size
                                gain = zeros_converted - ones_converted
                                max_gain = max(max_gain, gain)
                            
                            k = l
                        else:
                            k += 1
                
                i = j
            else:
                i += 1
        
        return original_count + max_gain