class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Augment the string with '1's at both ends
        t = '1' + s + '1'
        
        # Count the initial number of active sections
        initial_active_sections = t.count('1') - 1  # Subtract 1 for the augmented '1' at the start
        
        max_active_sections = initial_active_sections
        
        # Find blocks of '0's surrounded by '1's
        zero_blocks = []
        current_zero_length = 0
        
        for char in t:
            if char == '0':
                current_zero_length += 1
            else:
                if current_zero_length > 0:
                    zero_blocks.append(current_zero_length)
                current_zero_length = 0
        
        # Iterate through the zero blocks to find the maximum possible active sections after trade
        for i in range(len(zero_blocks)):
            # If we trade the current block of '0's
            # We lose the '1's on both sides of the block
            # But we gain the entire length of the block as '1's
            # The number of '1's we gain is equal to the length of the block
            # The number of '1's we lose is 2 (the two '1's surrounding the block)
            gain = zero_blocks[i]
            if i > 0:  # There is a '1' before the block
                gain -= 1  # We lose one '1' from the left side
            if i < len(zero_blocks) - 1:  # There is a '1' after the block
                gain -= 1  # We lose one '1' from the right side
            
            # Calculate the new number of active sections after this trade
            new_active_sections = initial_active_sections + gain
            
            # Update the maximum active sections found
            max_active_sections = max(max_active_sections, new_active_sections)
        
        return max_active_sections