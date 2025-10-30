class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_forbidden_len = max(len(f) for f in forbidden)
        n = len(word)
        
        def is_valid(start, end):
            for i in range(start, end + 1):
                for j in range(i, min(i + max_forbidden_len, end + 1)):
                    if word[i:j+1] in forbidden_set:
                        return False
            return True
        
        left = 0
        max_length = 0
        
        for right in range(n):
            while not is_valid(left, right):
                left += 1
            max_length = max(max_length, right - left + 1)
        
        return max_length