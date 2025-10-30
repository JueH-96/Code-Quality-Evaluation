class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            partitions = 0
            while s:
                chars = set()
                for i, char in enumerate(s):
                    chars.add(char)
                    if len(chars) > k:
                        s = s[i:]
                        break
                else:
                    s = ""
                partitions += 1
            return partitions

        max_partitions = 0
        max_partitions = max(max_partitions, count_partitions(s, k))
        
        for i in range(len(s)):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                new_char = chr(char_code)
                if new_char != original_char:
                    temp_s = list(s)
                    temp_s[i] = new_char
                    temp_s = "".join(temp_s)
                    max_partitions = max(max_partitions, count_partitions(temp_s, k))
        
        return max_partitions