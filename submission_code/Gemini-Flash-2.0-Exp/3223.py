class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                substring = word[i:i + length]
                
                char_counts = {}
                is_complete = True
                
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char in char_counts:
                    if char_counts[char] != k:
                        is_complete = False
                        break
                
                if not is_complete:
                    continue
                
                for j in range(len(substring) - 1):
                    if abs(ord(substring[j]) - ord(substring[j+1])) > 2:
                        is_complete = False
                        break
                
                if is_complete:
                    count += 1
        
        return count