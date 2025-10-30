from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # dp maps (first_char, last_char) to minimal length of the built string so far
        # Initialize with the first word
        first, last = words[0][0], words[0][-1]
        dp = {(first, last): len(words[0])}
        
        # Process each subsequent word
        for w in words[1:]:
            fw, lw = w[0], w[-1]
            wlen = len(w)
            new_dp = {}
            
            # For each previous state (f, l) with length cur_len
            for (f, l), cur_len in dp.items():
                # 1) join(str_prev, w)
                # If l == fw, we save 1 char
                cost1 = cur_len + wlen - (1 if l == fw else 0)
                key1 = (f, lw)
                if key1 not in new_dp or cost1 < new_dp[key1]:
                    new_dp[key1] = cost1
                
                # 2) join(w, str_prev)
                # If lw == f, we save 1 char
                cost2 = wlen + cur_len - (1 if lw == f else 0)
                key2 = (fw, l)
                if key2 not in new_dp or cost2 < new_dp[key2]:
                    new_dp[key2] = cost2
            
            dp = new_dp
        
        # The answer is the minimal length among all end states
        return min(dp.values())