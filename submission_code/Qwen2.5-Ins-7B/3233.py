class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_distinct_chars(s):
            return len(set(s))
        
        def find_longest_prefix(s, k):
            distinct_count = 0
            prefix = ""
            for i, char in enumerate(s):
                if count_distinct_chars(prefix + char) <= k:
                    prefix += char
                else:
                    return prefix
            return prefix
        
        def max_partitions(s, k):
            partitions = 0
            while s:
                prefix = find_longest_prefix(s, k)
                s = s[len(prefix):]
                partitions += 1
            return partitions
        
        original_partitions = max_partitions(s, k)
        max_possible_partitions = original_partitions
        for i in range(len(s)):
            for new_char in 'abcdefghijklmnopqrstuvwxyz':
                if new_char != s[i]:
                    new_s = s[:i] + new_char + s[i+1:]
                    max_possible_partitions = max(max_possible_partitions, max_partitions(new_s, k))
        
        return max_possible_partitions