class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(string, k):
            count = 0
            n = len(string)
            i = 0
            while i < n:
                count += 1
                j = i
                distinct_chars = set()
                while j < n and len(distinct_chars) <= k:
                    distinct_chars.add(string[j])
                    if len(distinct_chars) <= k:
                        j += 1
                i = j
            return count

        n = len(s)
        ans = 0
        for i in range(n):
            for char_code in range(ord('a'), ord('z') + 1):
                new_s = list(s)
                new_s[i] = chr(char_code)
                new_s = "".join(new_s)
                ans = max(ans, count_partitions(new_s, k))

        ans = max(ans, count_partitions(s, k))
        return ans