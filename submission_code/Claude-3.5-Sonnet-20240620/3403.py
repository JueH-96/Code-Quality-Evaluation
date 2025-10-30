class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        char_count = {}
        distinct_chars = 0
        partitions = 0
        
        for i, char in enumerate(s):
            if char not in char_count:
                char_count[char] = 0
                distinct_chars += 1
            
            char_count[char] += 1
            
            if char_count[char] == 1:
                distinct_chars += 1
            elif char_count[char] == 2:
                distinct_chars -= 1
            
            if distinct_chars == 0:
                partitions += 1
                char_count.clear()
                distinct_chars = 0
        
        if distinct_chars > 0:
            partitions += 1
        
        return partitions