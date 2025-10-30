class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        
        @lru_cache(None)
        def solve(idx, last_char):
            if idx == n:
                return 0
            
            word = words[idx]
            
            # Option 1: join(str_i-1, words[i])
            if last_char == word[0]:
                len1 = len(word) - 1 + solve(idx + 1, word[-1])
            else:
                len1 = len(word) + solve(idx + 1, word[-1])
                
            # Option 2: join(words[i], str_i-1)
            if word[-1] == last_char:
                len2 = len(word) - 1 + solve(idx + 1, word[0])
            else:
                len2 = len(word) + solve(idx + 1, word[0])
            
            return min(len1, len2)
        
        return len(words[0]) + solve(1, words[0][-1])