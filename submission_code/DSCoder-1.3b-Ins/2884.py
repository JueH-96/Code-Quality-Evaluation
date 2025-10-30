from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        left, right = 0, len(word) - 1
        max_length = 0

        while left <= right:
            for i in range(right, left - 1, -1):
                if word[i] in forbidden_set:
                    right = i - 1
                    break
            else:
                max_length = max(max_length, right - left + 1)
                left += 1
                right = left if left == len(word) else right

        return max_length