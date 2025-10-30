class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        count = [0] * 26
        partitions = 0
        temp_count = [0] * 26
        
        for char in s:
            temp_count[ord(char) - ord('a')] += 1
            
            if all(temp_count[i] == temp_count[0] for i in range(26)):
                partitions += 1
                temp_count = [0] * 26
        
        return partitions