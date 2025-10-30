class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # We'll use dynamic programming to keep track of all possible states
        # of (first_char, last_char) of the current concatenated string, along
        # with the minimum length needed to attain that state. 
        #
        # Each state is dp[c1][c2] = minimum length of a string whose first character
        # is c1 and last character is c2 after processing some number of words.
        #
        # When we add a new word w, we can do either:
        #   new_str = join(current_str, w)
        #     => cost increases by len(w) - (1 if current_str ends == w starts else 0)
        #   new_str = join(w, current_str)
        #     => cost increases by len(w) - (1 if w ends == current_str starts else 0)
        #
        # We track the new first/last characters and update dp accordingly.
        
        n = len(words)
        # Precompute for each word: first_char, last_char, length
        word_info = []
        for w in words:
            word_info.append((w[0], w[-1], len(w)))
        
        # Number of possible lowercase letters
        ALPH = 26
        
        # Convert character to index 0..25
        def idx(c):
            return ord(c) - ord('a')
        
        # Initialize dp array with infinities
        import math
        INF = math.inf
        
        # dp[fc][lc] will hold the minimal length of a string
        # whose first char is fc, last char is lc
        dp = [[INF]*ALPH for _ in range(ALPH)]
        
        # Initialize with the first word
        f, l, length = word_info[0]
        dp[idx(f)][idx(l)] = length
        
        # Iterate through the remaining words
        for i in range(1, n):
            nf, nl, wlen = word_info[i]
            new_dp = [[INF]*ALPH for _ in range(ALPH)]
            for fc in range(ALPH):
                for lc in range(ALPH):
                    curr_len = dp[fc][lc]
                    if curr_len == INF:
                        continue
                    # Option 1: join(str, w)
                    # If the last char of str == first char of w, reduce by 1
                    overlap1 = 1 if lc == idx(nf) else 0
                    length1 = curr_len + wlen - overlap1
                    new_dp[fc][idx(nl)] = min(new_dp[fc][idx(nl)], length1)
                    
                    # Option 2: join(w, str)
                    # If the last char of w == first char of str, reduce by 1
                    overlap2 = 1 if idx(nl) == fc else 0
                    length2 = curr_len + wlen - overlap2
                    new_dp[idx(nf)][lc] = min(new_dp[idx(nf)][lc], length2)
            dp = new_dp
        
        # The answer is the minimum length among all possible first/last chars
        ans = min(min(row) for row in dp)
        return ans