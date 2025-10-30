class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        from collections import defaultdict

        def max_partitions(s):
            partitions = 0
            start = 0
            while start < len(s):
                char_count = defaultdict(int)
                distinct_count = 0
                end = start
                while end < len(s) and (distinct_count < k or (distinct_count == k and char_count[s[end]] > 0)):
                    if char_count[s[end]] == 0:
                        distinct_count += 1
                    char_count[s[end]] += 1
                    end += 1
                partitions += 1
                start = end
            return partitions

        # Calculate partitions without any change
        max_partitions_count = max_partitions(s)

        # Try changing each character and calculate partitions
        for i in range(len(s)):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != original_char:
                    new_s = s[:i] + c + s[i+1:]
                    max_partitions_count = max(max_partitions_count, max_partitions(new_s))

        return max_partitions_count