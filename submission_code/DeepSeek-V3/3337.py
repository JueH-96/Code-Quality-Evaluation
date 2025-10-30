class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        for char in s:
            if char == c:
                count += 1
        # The number of substrings is the sum of the first 'count' natural numbers
        return count * (count + 1) // 2