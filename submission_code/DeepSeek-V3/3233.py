class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        if k >= 26:
            return n
        
        def count_partitions(s):
            partitions = 0
            i = 0
            while i < n:
                unique = set()
                j = i
                while j < n and len(unique) <= k:
                    unique.add(s[j])
                    if len(unique) > k:
                        break
                    j += 1
                partitions += 1
                i = j
            return partitions
        
        max_partitions = count_partitions(s)
        
        for i in range(n):
            original_char = s[i]
            for new_char in 'abcdefghijklmnopqrstuvwxyz':
                if new_char == original_char:
                    continue
                new_s = s[:i] + new_char + s[i+1:]
                current_partitions = count_partitions(new_s)
                if current_partitions > max_partitions:
                    max_partitions = current_partitions
            if max_partitions == n:
                break
        
        return max_partitions