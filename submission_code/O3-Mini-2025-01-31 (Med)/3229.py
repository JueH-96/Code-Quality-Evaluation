from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Helper function: given an integer r, generate a set of candidate palindrome numbers
        # that are "near" r by modifying the prefix.
        def getPalCandidates(r: int) -> set:
            s = str(r)
            d = len(s)
            half = (d + 1) // 2  # length of the left part used for reflection
            prefix = int(s[:half])
            candidates = set()
            
            # try prefix modifications
            for diff in [-1, 0, 1]:
                new_prefix = prefix + diff
                new_prefix_str = str(new_prefix)
                # The new prefix might have lost a digit (e.g., when r=1000 and diff=-1 changes prefix '10' to '9')
                if len(new_prefix_str) < half:
                    # Candidate: all 9's with length d-1 (if d > 1)
                    if d > 1:
                        candidates.add(int("9" * (d - 1)))
                # The new prefix could be longer (e.g., when r=999 and diff=1 leads to prefix 10...),
                # then candidate is 10^d + 1 (which is e.g., 1001 for d=3)
                elif len(new_prefix_str) > half:
                    candidates.add(int("1" + "0" * (d - 1) + "1"))
                else:
                    # Build the palindrome based on whether d is even or odd
                    if d % 2 == 0:
                        pal = new_prefix_str + new_prefix_str[::-1]
                    else:
                        pal = new_prefix_str + new_prefix_str[:-1][::-1]
                    candidates.add(int(pal))
            
            # Also add the natural boundary candidates
            if d > 1:
                candidates.add(int("9" * (d - 1)))
            candidates.add(int("1" + "0" * (d - 1) + "1"))
            
            # Ensure that the candidates are valid (positive and less than 10^9)
            candidates = {cand for cand in candidates if cand >= 1 and cand < 10**9}
            
            return candidates
        
        # For sum(|x - y|) the best unconstrained y is the median 
        n = len(nums)
        nums_sorted = sorted(nums)
        if n % 2 == 1:
            median = nums_sorted[n // 2]
            # For union, also include the median itself if even case arises
            candidate_bases = [median]
        else:
            # When even, any value between the two medians minimizes the sum.
            m1 = nums_sorted[n // 2 - 1]
            m2 = nums_sorted[n // 2]
            # We can take the floor of their average (or both medians) as bases
            median = (m1 + m2) // 2
            candidate_bases = [m1, m2, median]
        
        candidate_set = set()
        for base in candidate_bases:
            candidate_set |= getPalCandidates(base)
        
        # Now choose the candidate palindrome y that minimizes the total cost = sum(|xi - y|)
        best_cost = float('inf')
        for candidate in candidate_set:
            # It is safe to use sum() over a generator; n is at most 1e5 so this is efficient
            total_cost = sum(abs(x - candidate) for x in nums)
            if total_cost < best_cost:
                best_cost = total_cost
        
        return best_cost