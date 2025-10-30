class Solution:
    def maxOperations(self, s: str) -> int:
        count_ones = 0
        result = 0
        n = len(s)
        i = 0
        while i < n:
            if s[i] == '1':
                count_ones += 1
                i += 1
            else:
                # Start of a zero group
                result += count_ones
                j = i
                while j < n and s[j] == '0':
                    j += 1
                i = j
        return result