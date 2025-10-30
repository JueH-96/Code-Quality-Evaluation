class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            partitions = 0
            start = 0
            while start < len(s):
                end = start
                distinct_chars = set()
                while end < len(s) and (len(distinct_chars) < k or s[end] in distinct_chars):
                    distinct_chars.add(s[end])
                    end += 1
                partitions += 1
                start = end
            return partitions
        
        max_partitions = count_partitions(s, k)
        for i in range(len(s)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if s[i] != c:
                    new_s = s[:i] + c + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(new_s, k))
        return max_partitions