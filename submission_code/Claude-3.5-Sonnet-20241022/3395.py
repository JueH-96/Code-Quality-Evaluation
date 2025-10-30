class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        for i in range(1, n + 1):
            if n % i == 0:
                # Check if string can be divided into anagrams of length i
                freq = {}
                valid = True
                
                # Get frequency of first substring of length i
                for j in range(i):
                    freq[s[j]] = freq.get(s[j], 0) + 1
                
                # Check all other substrings of length i
                for j in range(i, n, i):
                    curr_freq = {}
                    # Get frequency of current substring
                    for k in range(j, j + i):
                        curr_freq[s[k]] = curr_freq.get(s[k], 0) + 1
                    
                    # Compare with first substring frequency
                    if curr_freq != freq:
                        valid = False
                        break
                
                if valid:
                    return i
        
        return n