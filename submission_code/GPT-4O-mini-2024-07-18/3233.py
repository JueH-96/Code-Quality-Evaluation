class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def max_partitions(s: str, k: int) -> int:
            count = {}
            distinct_count = 0
            partitions = 0
            i = 0
            
            while i < len(s):
                while i < len(s) and (distinct_count < k or s[i] in count):
                    if s[i] not in count:
                        count[s[i]] = 0
                        distinct_count += 1
                    count[s[i]] += 1
                    i += 1
                partitions += 1
                for char in count:
                    count[char] = 0
                distinct_count = 0
            
            return partitions
        
        max_partitions = max_partitions(s, k)
        
        for i in range(len(s)):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != original_char:
                    modified_s = s[:i] + c + s[i+1:]
                    max_partitions = max(max_partitions, max_partitions(modified_s, k))
        
        return max_partitions