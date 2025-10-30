class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        from collections import Counter
        
        def is_complete(s):
            count = Counter(s)
            for char in count:
                if count[char] % k != 0:
                    return False
            return True
        
        n = len(word)
        count = 0
        for i in range(n):
            if i > 0 and abs(ord(word[i]) - ord(word[i-1])) > 2:
                continue
            for j in range(i, n):
                if j > i and abs(ord(word[j]) - ord(word[j-1])) > 2:
                    break
                if is_complete(word[i:j+1]):
                    count += 1
        return count