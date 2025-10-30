class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        from collections import defaultdict
        
        def count_partitions(s, k):
            n = len(s)
            i = 0
            partitions = 0
            while i < n:
                distinct_count = 0
                char_count = defaultdict(int)
                j = i
                while j < n and (distinct_count < k or (distinct_count == k and s[j] in char_count)):
                    if char_count[s[j]] == 0:
                        distinct_count += 1
                    char_count[s[j]] += 1
                    j += 1
                partitions += 1
                i = j
            return partitions
        
        # Check partitions without any changes
        max_partitions = count_partitions(s, k)
        
        # Try changing each character to any other character and calculate partitions
        for i in range(len(s)):
            original_char = s[i]
            for new_char in 'abcdefghijklmnopqrstuvwxyz':
                if new_char != original_char:
                    modified_s = s[:i] + new_char + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(modified_s, k))
        
        return max_partitions