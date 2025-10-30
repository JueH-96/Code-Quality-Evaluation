class Solution:
    def count_partitions(self, s: str, k: int) -> int:
        partitions = 0
        current_s = s
        while current_s:
            longest_prefix = ""
            for i in range(1, len(current_s) + 1):
                prefix = current_s[:i]
                distinct_chars = set(prefix)
                if len(distinct_chars) <= k:
                    longest_prefix = prefix
                else:
                    break
            if not longest_prefix:
                longest_prefix = current_s[0]
            current_s = current_s[len(longest_prefix):]
            partitions += 1
        return partitions

    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        initial_partitions = self.count_partitions(s, k)
        max_partitions = initial_partitions
        n = len(s)
        for i in range(n):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                temp_s_list = list(s)
                temp_s_list[i] = char
                temp_s = "".join(temp_s_list)
                current_partitions = self.count_partitions(temp_s, k)
                max_partitions = max(max_partitions, current_partitions)
        return max_partitions