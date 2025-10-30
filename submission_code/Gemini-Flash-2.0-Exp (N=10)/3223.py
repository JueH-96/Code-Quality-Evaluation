class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = word[i:j+1]
                if len(sub) == 0:
                    continue
                
                char_counts = {}
                valid_counts = True
                for char in sub:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char in char_counts:
                    if char_counts[char] != k:
                        valid_counts = False
                        break
                
                if not valid_counts:
                    continue
                
                valid_diff = True
                for l in range(len(sub) - 1):
                    if abs(ord(sub[l]) - ord(sub[l+1])) > 2:
                        valid_diff = False
                        break
                
                if valid_diff:
                    count += 1
        return count