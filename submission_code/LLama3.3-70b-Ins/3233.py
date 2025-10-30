class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s: str, k: int) -> int:
            partitions = 0
            while s:
                i = 0
                distinct_chars = set()
                while i < len(s) and len(distinct_chars) <= k:
                    distinct_chars.add(s[i])
                    i += 1
                s = s[i:]
                partitions += 1
            return partitions

        max_partitions = 0
        for i in range(len(s)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_s = s[:i] + char + s[i+1:]
                max_partitions = max(max_partitions, count_partitions(new_s, k))
        max_partitions = max(max_partitions, count_partitions(s, k))
        return max_partitions