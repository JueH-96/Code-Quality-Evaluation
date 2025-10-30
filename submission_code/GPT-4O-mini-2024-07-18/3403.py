class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import defaultdict
        
        count = defaultdict(int)
        partitions = 0
        
        for char in s:
            count[char] += 1
            
            # Check if all characters have the same frequency
            if len(set(count.values())) == 1:
                partitions += 1
                count.clear()  # Reset the count for the next substring
        
        return partitions