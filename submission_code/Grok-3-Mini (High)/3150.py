class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        for i in range(n):
            ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones += 1
                if ones == k:
                    length = j - i + 1
                    if length < min_len:
                        min_len = length
                        result = s[i:j+1]
                    elif length == min_len:
                        if s[i:j+1] < result:
                            result = s[i:j+1]
                    break  # Stop after finding the first substring with exactly k ones for this start index
                elif ones > k:
                    break  # No need to continue if ones exceed k
        return result