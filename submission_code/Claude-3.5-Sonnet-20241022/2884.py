class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_forbidden_len = max(len(f) for f in forbidden)
        n = len(word)
        
        def isValid(start, end):
            curr = end - start + 1
            # Check all possible substrings ending at 'end'
            for length in range(1, min(curr, max_forbidden_len) + 1):
                if word[end-length+1:end+1] in forbidden_set:
                    return False
            return True
        
        ans = 0
        left = 0
        
        # For each right pointer
        for right in range(n):
            # While the substring is invalid, move left pointer
            while left <= right and not isValid(left, right):
                left += 1
            
            # Update answer with current valid window size
            ans = max(ans, right - left + 1)
            
        return ans