class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s: str, k: int) -> int:
            partitions = 0
            i = 0
            while i < len(s):
                distinct_chars = set()
                j = i
                while j < len(s) and len(distinct_chars) <= k:
                    distinct_chars.add(s[j])
                    j += 1
                if len(distinct_chars) > k:
                    j -= 1
                partitions += 1
                i = j
            return partitions

        max_partitions = count_partitions(s, k)

        for i in range(len(s)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if s[i] != c:
                    new_s = s[:i] + c + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(new_s, k))

        return max_partitions