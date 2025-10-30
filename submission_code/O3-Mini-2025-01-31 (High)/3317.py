from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # First compute global frequency of all letters 
        total_freq = Counter()
        for word in words:
            total_freq.update(word)
        # TotalPairs: from each letter, how many pairs can be formed
        totalPairs = sum(count // 2 for count in total_freq.values())
        # TotalSingles: free singles (leftovers) in global pool
        totalSingles = sum(count % 2 for count in total_freq.values())
        
        even_costs = []  # for even-length words, cost = len(w)//2.
        odd_costs = []   # for odd-length words, base cost = (len(w)-1)//2.
        for w in words:
            L = len(w)
            if L % 2 == 0:
                even_costs.append(L // 2)
            else:
                odd_costs.append((L - 1) // 2)
        
        even_costs.sort()
        odd_costs.sort()
        
        m_even = len(even_costs)
        m_odd = len(odd_costs)
        
        # Precompute prefix sums for even and odd cost lists.
        # prefix_even[i] = sum of even_costs[0...i-1]
        prefix_even = [0]
        for cost in even_costs:
            prefix_even.append(prefix_even[-1] + cost)
            
        prefix_odd = [0]
        for cost in odd_costs:
            prefix_odd.append(prefix_odd[-1] + cost)
            
        n = len(words)
        best = 0
        # We try to choose m words (0 <= m <= n) that we want to fix to palindromes.
        # Among these m words, let j be the number of odd–length words chosen.
        # Then, (m - j) will be from the even–length group.
        # Note: in an odd word, if it uses a "coupon" (up to totalSingles available) then its cost is its base cost.
        # Otherwise, it pays an extra 1 (i.e. cost = base cost + 1).
        # So for j odd words the extra penalty is max(0, j - totalSingles).
        for m in range(n + 1):
            # number of odd words chosen j can range:
            lo = max(0, m - m_even)  # if m-m_even even words must come from even group, j must be at least m - (#evens available)
            hi = min(m, m_odd)       # can't choose more odd words than m or than available
            feasible = False
            # Try all splits: choose j odd words and (m - j) even words.
            for j in range(lo, hi + 1):
                cost_even = prefix_even[m - j]     # cost for the (m - j) smallest even words
                cost_odd = prefix_odd[j]            # cost for the j smallest odd words
                extra_penalty = max(0, j - totalSingles)
                total_cost = cost_even + cost_odd + extra_penalty
                if total_cost <= totalPairs:
                    feasible = True
                    break
            if feasible:
                best = m
        return best