class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            partitions = 0
            while s:
                distinct_chars = set()
                length = 0
                for i in range(len(s)):
                    distinct_chars.add(s[i])
                    if len(distinct_chars) > k:
                        break
                    length += 1
                s = s[length:]
                partitions += 1
            return partitions

        n = len(s)
        max_partitions = count_partitions(s, k)

        for i in range(n):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                new_char = chr(char_code)
                temp_s = list(s)
                temp_s[i] = new_char
                temp_s = "".join(temp_s)
                max_partitions = max(max_partitions, count_partitions(temp_s, k))

        return max_partitions