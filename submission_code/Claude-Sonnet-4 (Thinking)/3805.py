class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        augmented = '1' + s + '1'
        
        def find_ones_blocks(string):
            blocks = []
            i = 1
            while i < len(string) - 1:
                if string[i] == '1':
                    start = i
                    while i < len(string) - 1 and string[i] == '1':
                        i += 1
                    end = i - 1
                    if string[start - 1] == '0' and string[end + 1] == '0':
                        blocks.append((start, end))
                else:
                    i += 1
            return blocks
        
        def find_zeros_blocks(string):
            blocks = []
            i = 1
            while i < len(string) - 1:
                if string[i] == '0':
                    start = i
                    while i < len(string) - 1 and string[i] == '0':
                        i += 1
                    end = i - 1
                    if string[start - 1] == '1' and string[end + 1] == '1':
                        blocks.append((start, end))
                else:
                    i += 1
            return blocks
        
        original_count = s.count('1')
        max_count = original_count
        
        ones_blocks = find_ones_blocks(augmented)
        
        for start1, end1 in ones_blocks:
            # Convert this block of '1's to '0's
            temp_augmented = list(augmented)
            for i in range(start1, end1 + 1):
                temp_augmented[i] = '0'
            temp_augmented = ''.join(temp_augmented)
            
            zeros_blocks = find_zeros_blocks(temp_augmented)
            
            for start2, end2 in zeros_blocks:
                # Calculate the final count
                temp_count = temp_augmented[1:-1].count('1')  # Count '1's in modified string (excluding augmented chars)
                temp_count += (end2 - start2 + 1)  # Add the '1's from converting the '0' block
                max_count = max(max_count, temp_count)
        
        return max_count