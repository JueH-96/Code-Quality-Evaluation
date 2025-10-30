class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(current_s, current_k):
            partitions = 0
            temp_s = list(current_s)
            while temp_s:
                longest_prefix_len = 0
                for prefix_len in range(1, len(temp_s) + 1):
                    prefix = "".join(temp_s[:prefix_len])
                    distinct_chars = set(prefix)
                    if len(distinct_chars) <= current_k:
                        longest_prefix_len = prefix_len
                    else:
                        break
                if longest_prefix_len == 0:
                    break # Should not happen given k >= 1 and string is not empty
                temp_s = temp_s[longest_prefix_len:]
                partitions += 1
            return partitions

        original_partitions = count_partitions(s, k)
        max_partitions = original_partitions
        s_list = list(s)
        for i in range(len(s)):
            original_char = s_list[i]
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                s_list[i] = char
                modified_s = "".join(s_list)
                current_partitions = count_partitions(modified_s, k)
                max_partitions = max(max_partitions, current_partitions)
            s_list[i] = original_char # backtrack

        return max_partitions