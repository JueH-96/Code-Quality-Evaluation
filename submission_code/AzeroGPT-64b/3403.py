from collections import Counter

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        char_counts = Counter(s)
        min_partitions = 1
        
        current_partition_count = 0
        current_counts = Counter()
        
        for char in s:
            current_counts[char] += 1
            if current_counts[char] == char_counts[char]:
                current_partition_count += 1
                
            if current_partition_count == len(char_counts):
                current_counts.clear()
                current_partition_count = 0
                min_partitions += 1
        
        return min_partitions if min_partitions > 1 else 1