class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from math import inf
        
        # Precompute for each word: first char index, last char index, length
        n = len(words)
        flen = []
        for w in words:
            f = ord(w[0]) - ord('a')
            l = ord(w[-1]) - ord('a')
            flen.append((f, l, len(w)))
        
        # If there's only one word, answer is simply its length
        if n == 1:
            return flen[0][2]
        
        # dpPrev[f][l] will store the minimum length of the constructed string
        # after processing some number of words, if its first character is f
        # and its last character is l.
        dpPrev = [[inf]*26 for _ in range(26)]
        
        # Initialize with the first word
        f0, l0, len0 = flen[0]
        dpPrev[f0][l0] = len0
        
        # Process words from index 1 to n-1
        for i in range(1, n):
            fi, li, wi = flen[i]
            dpCurr = [[inf]*26 for _ in range(26)]
            
            for f in range(26):
                for l in range(26):
                    prev_len = dpPrev[f][l]
                    if prev_len == inf:
                        continue
                    
                    # Option 1: join(str_{i-1}, words[i])
                    # If the last char of str_{i-1} == first char of words[i], we save 1 char
                    new_len = prev_len + wi - (1 if l == fi else 0)
                    dpCurr[f][li] = min(dpCurr[f][li], new_len)
                    
                    # Option 2: join(words[i], str_{i-1})
                    # If the last char of words[i] == first char of str_{i-1}, we save 1 char
                    new_len = prev_len + wi - (1 if li == f else 0)
                    dpCurr[fi][l] = min(dpCurr[fi][l], new_len)
            
            dpPrev = dpCurr
        
        # The answer is the minimum over all possible first and last characters
        return min(dpPrev[f][l] for f in range(26) for l in range(26))