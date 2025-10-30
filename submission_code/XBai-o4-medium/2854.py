from typing import List
from collections import defaultdict

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize DP with the first word's first and last characters and its length
        first = words[0]
        f0 = first[0]
        l0 = first[-1]
        dp = {(f0, l0): len(first)}
        
        for i in range(1, len(words)):
            word = words[i]
            new_f = word[0]
            new_l = word[-1]
            new_len = len(word)
            next_dp = {}
            
            for (current_f, current_l), current_total in dp.items():
                # Option 1: append the new word to the current string
                total1 = current_total + new_len
                if current_l == new_f:
                    total1 -= 1
                new_state1 = (current_f, new_l)
                if new_state1 in next_dp:
                    if total1 < next_dp[new_state1]:
                        next_dp[new_state1] = total1
                else:
                    next_dp[new_state1] = total1
                
                # Option 2: prepend the new word to the current string
                total2 = current_total + new_len
                if new_l == current_f:
                    total2 -= 1
                new_state2 = (new_f, current_l)
                if new_state2 in next_dp:
                    if total2 < next_dp[new_state2]:
                        next_dp[new_state2] = total2
                else:
                    next_dp[new_state2] = total2
            
            dp = next_dp
        
        return min(dp.values())