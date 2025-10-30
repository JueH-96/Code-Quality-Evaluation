class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            partitions = 0
            i = 0
            while i < len(s):
                distinct_chars = set()
                j = i
                while j < len(s) and len(distinct_chars) <= k:
                    distinct_chars.add(s[j])
                    if len(distinct_chars) > k:
                        break
                    j += 1
                partitions += 1
                i = j
            return partitions
        
        max_partitions = count_partitions(s, k)
        
        if k == 26:
            return max_partitions
        
        for i in range(len(s)):
            original_char = s[i]
            for new_char in 'abcdefghijklmnopqrstuvwxyz':
                if new_char == original_char:
                    continue
                modified_s = s[:i] + new_char + s[i+1:]
                max_partitions = max(max_partitions, count_partitions(modified_s, k))
        
        return max_partitions