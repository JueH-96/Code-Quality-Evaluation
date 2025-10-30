from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Helper: Given a prefix number and total length L of the palindrome we want,
        # construct a palindrome with that prefix.
        def make_palindrome(prefix: int, L: int) -> int:
            p_str = str(prefix)
            # For even L, we expect the prefix length to be L//2;
            # for odd L, the prefix length should be (L+1)//2.
            if L % 2 == 0:
                return int(p_str + p_str[::-1])
            else:
                return int(p_str + p_str[:-1][::-1])
        
        n = len(nums)
        # The sum of absolute differences is minimized when choosing a value in the median interval.
        # We choose one median as our anchor.
        nums_sorted = sorted(nums)
        median = nums_sorted[n // 2]
        
        s = str(median)
        L = len(s)
        # Determine the "half" length.
        half_len = (L + 1) // 2 if (L % 2 == 1) else (L // 2)
        prefix_str = s[:half_len]
        prefix_num = int(prefix_str)
        
        candidates = set()
        
        # Candidate using the original prefix (this will equal the median if it's a palindrome).
        cand1 = make_palindrome(prefix_num, L)
        candidates.add(cand1)
        
        # Candidate by decrementing the prefix.
        # If prefix_num decreases and its digit count stays the same, use that.
        # Otherwise, add the special candidate: all 9's with one digit less.
        if prefix_num - 1 > 0 and len(str(prefix_num - 1)) == len(prefix_str):
            candidates.add(make_palindrome(prefix_num - 1, L))
        else:
            if L > 1:
                candidates.add(10 ** (L - 1) - 1)
        
        # Candidate by incrementing the prefix.
        # If the new prefix does not increase digit length, use it; otherwise use special candidate.
        if len(str(prefix_num + 1)) == len(prefix_str):
            candidates.add(make_palindrome(prefix_num + 1, L))
        else:
            # The special candidate for an increased digit count:
            if 10 ** L + 1 < 10 ** 9:
                candidates.add(10 ** L + 1)
        
        # Also add the special candidates:
        # Candidate: all 9's with one digit less (if L > 1).
        if L > 1:
            candidates.add(10 ** (L - 1) - 1)
        # Candidate: smallest palindrome with one digit more; only if it's less than 10^9.
        if L < 9 and (10 ** L + 1) < 10 ** 9:
            candidates.add(10 ** L + 1)
        
        # Filter candidates to ensure they are positive and less than 10^9.
        valid_candidates = [p for p in candidates if p > 0 and p < 10 ** 9]
        
        best_cost = float('inf')
        # For each candidate palindrome, compute the total cost (sum of absolute differences).
        for candidate in valid_candidates:
            cost = sum(abs(num - candidate) for num in nums)
            best_cost = min(best_cost, cost)
            
        return best_cost