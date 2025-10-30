class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(string, k):
            partitions = 0
            while string:
                partitions += 1
                distinct_chars = set()
                max_len = 0
                for i in range(len(string)):
                    distinct_chars.add(string[i])
                    if len(distinct_chars) <= k:
                        max_len += 1
                    else:
                        break
                string = string[max_len:]
            return partitions

        max_partitions = 0
        for i in range(len(s) + 1):
            for char_code in range(ord('a'), ord('z') + 1):
                temp_s = list(s)
                if i < len(s):
                    temp_s[i] = chr(char_code)
                max_partitions = max(max_partitions, count_partitions("".join(temp_s), k))

        return max_partitions