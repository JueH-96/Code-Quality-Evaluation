class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        from collections import Counter
        
        def count_partitions(s, k):
            partitions = 0
            i = 0
            while i < len(s):
                distinct_chars = set()
                while i < len(s) and (len(distinct_chars) < k or s[i] in distinct_chars):
                    distinct_chars.add(s[i])
                    i += 1
                partitions += 1
            return partitions
        
        max_partitions = count_partitions(s, k)
        
        for i in range(len(s)):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != original_char:
                    new_s = s[:i] + c + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(new_s, k))
        
        return max_partitions