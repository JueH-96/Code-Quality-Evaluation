class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Original count of active sections
        original_count = s.count('1')
        max_active = original_count
        
        # Augmented string
        t = '1' + s + '1'
        
        # Find all blocks of 1s surrounded by 0s
        blocks_of_1s = []
        i = 1  # Start from index 1 to skip the augmented '1' at the beginning
        while i < len(t) - 1:
            if t[i] == '1':
                start = i
                while i < len(t) - 1 and t[i] == '1':
                    i += 1
                # Check if the block is surrounded by 0s
                if start > 0 and i < len(t) and t[start-1] == '0' and t[i] == '0':
                    blocks_of_1s.append((start, i-1))
            else:
                i += 1
        
        # Simulate each possible trade
        for block in blocks_of_1s:
            start, end = block
            # Convert the block from 1s to 0s
            new_t = t[:start] + '0' * (end - start + 1) + t[end+1:]
            
            # Find all blocks of 0s surrounded by 1s in the new string
            blocks_of_0s = []
            j = 1
            while j < len(new_t) - 1:
                if new_t[j] == '0':
                    start_0 = j
                    while j < len(new_t) - 1 and new_t[j] == '0':
                        j += 1
                    # Check if the block is surrounded by 1s
                    if start_0 > 0 and j < len(new_t) and new_t[start_0-1] == '1' and new_t[j] == '1':
                        blocks_of_0s.append((start_0, j-1))
                else:
                    j += 1
            
            # Try converting each valid block of 0s to 1s
            for block_0 in blocks_of_0s:
                start_0, end_0 = block_0
                # Convert the block from 0s to 1s
                final_t = new_t[:start_0] + '1' * (end_0 - start_0 + 1) + new_t[end_0+1:]
                # Count active sections without augmentation
                active_sections = final_t[1:-1].count('1')
                max_active = max(max_active, active_sections)
        
        return max_active