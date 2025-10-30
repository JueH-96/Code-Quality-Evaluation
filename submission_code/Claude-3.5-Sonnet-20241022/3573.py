class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        w2_freq = {}
        for c in word2:
            w2_freq[c] = w2_freq.get(c, 0) + 1
        
        count = 0
        for i in range(n):
            curr_freq = {}
            for j in range(i, n):
                curr_freq[word1[j]] = curr_freq.get(word1[j], 0) + 1
                
                # Check if current substring can be rearranged to have word2 as prefix
                valid = True
                for c, freq in w2_freq.items():
                    if c not in curr_freq or curr_freq[c] < freq:
                        valid = False
                        break
                
                if valid:
                    count += 1
                    
        return count