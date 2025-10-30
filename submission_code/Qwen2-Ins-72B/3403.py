class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        char_count = {}
        partitions = 1
        
        for i, char in enumerate(s):
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            
            # Check if the current substring is balanced
            if all(count == char_count[list(char_count.keys())[0]] for count in char_count.values()):
                # Reset the character count for the next substring
                char_count = {}
                partitions += 1
        
        # Subtract 1 because we count an extra partition at the end
        return partitions - 1