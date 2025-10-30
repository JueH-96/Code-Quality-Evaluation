from typing import List
from collections import defaultdict

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # Initialize DP with the first word
        f = words[0][0]
        l = words[0][-1]
        length = len(words[0])
        dp = defaultdict(int)
        dp[(f, l)] = length
        
        # Iterate through the remaining words
        for word in words[1:]:
            new_dp = defaultdict(lambda: float('inf'))
            for (f_prev, l_prev), current_length in dp.items():
                # Option 1: append word to the right
                if l_prev == word[0]:
                    new_length = current_length + len(word) - 1
                else:
                    new_length = current_length + len(word)
                new_l = word[-1]
                new_f = f_prev
                new_dp[(new_f, new_l)] = min(new_dp[(new_f, new_l)], new_length)
                
                # Option 2: prepend word to the left
                if f_prev == word[-1]:
                    new_length = current_length + len(word) - 1
                else:
                    new_length = current_length + len(word)
                new_f = word[0]
                new_l = l_prev
                new_dp[(new_f, new_l)] = min(new_dp[(new_f, new_l)], new_length)
            dp = new_dp
        
        # Find the minimum length from the DP states
        return min(dp.values())