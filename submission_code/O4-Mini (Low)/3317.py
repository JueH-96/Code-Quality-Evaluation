from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count total characters
        cnt = Counter()
        for w in words:
            cnt.update(w)
        # Total pairs and singles available initially
        total_pairs = sum(v // 2 for v in cnt.values())
        singles = sum(v % 2 for v in cnt.values())
        
        # For each word compute (pairs_needed, singles_needed)
        reqs = []
        for w in words:
            L = len(w)
            p = L // 2
            s = L % 2
            reqs.append((p, s))
        # Sort by pairs needed ascending, and for same pairs, even-length (s=0) first
        reqs.sort(key=lambda x: (x[0], x[1]))
        
        P = total_pairs
        S = singles
        ans = 0
        
        for p_need, s_need in reqs:
            # Check if we can allocate pairs
            if P < p_need:
                break
            # Tentatively consume pairs
            P -= p_need
            
            # Now satisfy an odd center if needed
            if s_need == 1:
                if S >= 1:
                    S -= 1
                else:
                    # try to break one pair into two singles
                    if P >= 1:
                        P -= 1
                        S += 2
                        S -= 1
                    else:
                        # Can't fulfill odd center, rollback pairs and skip
                        P += p_need
                        continue
            # Successfully formed one palindrome
            ans += 1
        
        return ans