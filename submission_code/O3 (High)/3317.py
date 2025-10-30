from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # total counts of all characters
        total_cnt = Counter()
        for w in words:
            total_cnt.update(w)
        
        # how many pairs (two identical letters) and single letters we have
        pairs  = 0   # each value // 2
        singles = 0  # each value  % 2
        for c in total_cnt.values():
            pairs   += c // 2
            singles += c & 1          # same as c % 2
        
        # sort word lengths – we will try to build palindromes
        lengths = sorted(len(w) for w in words)
        
        answer = 0
        for L in lengths:
            need_pairs = L // 2          # pairs needed for the two symmetric halves
            if need_pairs > pairs:       # not enough pairs left – later words are longer ⇒ stop
                break
            pairs -= need_pairs          # reserve the required pairs
            
            if L & 1:                    # odd length → need one centre character
                if singles:              # have a free single letter
                    singles -= 1
                else:
                    # create 2 singles by breaking one pair
                    if pairs == 0:       # no pair to break ⇒ cannot build this palindrome
                        break
                    pairs   -= 1         # break one pair
                    singles += 1         # two singles produced, one used now, one left
            # palindrome built successfully
            answer += 1
        
        return answer