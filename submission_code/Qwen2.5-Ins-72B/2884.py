class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_length = 0
        n = len(word)
        right = n - 1
        
        for left in range(n - 1, -1, -1):
            while right > left and any(word[left:right+1].endswith(f) for f in forbidden_set):
                right -= 1
            max_length = max(max_length, right - left + 1)
        
        return max_length