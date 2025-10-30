class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(s):
            if len(s) % k != 0:
                return False
            count = {}
            for char in s:
                if char in count:
                    count[char] += 1
                else:
                    count[char] = 1
            for key in count:
                if count[key] != k:
                    return False
            for i in range(1, len(s)):
                if abs(ord(s[i]) - ord(s[i-1])) > 2:
                    return False
            return True
        
        n = len(word)
        result = 0
        for i in range(n):
            for j in range(i+1, min(i+26*k+1, n+1)):
                substring = word[i:j]
                if is_complete(substring):
                    result += 1
        return result