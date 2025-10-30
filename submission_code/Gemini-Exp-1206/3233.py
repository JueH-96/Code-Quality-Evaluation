class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def calculate_partitions(string, k):
            partitions = 0
            start = 0
            while start < len(string):
                partitions += 1
                end = start
                distinct_chars = set()
                while end < len(string):
                    distinct_chars.add(string[end])
                    if len(distinct_chars) > k:
                        break
                    end += 1
                start = end
            return partitions

        max_partitions = calculate_partitions(s, k)
        for i in range(len(s)):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                new_char = chr(char_code)
                if new_char != original_char:
                    new_s = s[:i] + new_char + s[i+1:]
                    max_partitions = max(max_partitions, calculate_partitions(new_s, k))
            
        return max_partitions