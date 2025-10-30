from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Helper: construct palindrome from the left half for given length L.
        def make_palindrome(half: str, L: int) -> int:
            # if L is even, mirror the entire half in reverse.
            # if odd, mirror all but last character.
            if L % 2 == 0:
                return int(half + half[::-1])
            else:
                return int(half + half[-2::-1])
        
        # Given an integer m, generate a set of palindrome candidates near m.
        def gen_candidates(m: int) -> set:
            s = str(m)
            L = len(s)
            candidates = set()
            
            # function to try generating candidate for a given half and length L
            def try_half(raw_half: int, L: int):
                # raw_half might have fewer digits than required if, for instance, m is small.
                # ensure it is of proper length (leading zeros allowed in the construction? Not really,
                # but if raw_half becomes 0 then candidate won't be valid).
                half_str = str(raw_half)
                # If the length of half_str is not equal to required, adjust with zeros in front if needed.
                need = (L + 1) // 2
                if len(half_str) < need:
                    half_str = half_str.zfill(need)
                if len(half_str) > need:
                    return  # skip this candidate as it changes the digit length
                p = make_palindrome(half_str, L)
                # Only consider positive integers less than 10^9 as asked.
                if 1 <= p < 10**9:
                    candidates.add(p)
            
            need = (L + 1) // 2
            half_val = int(s[:need])
            # Try candidate based on current half, and variations
            for delta in [-1, 0, 1]:
                try_half(half_val + delta, L)
            
            # Additionally, sometimes the best candidate might come from a palindrome of a different length.
            # For instance, if m is 100, the optimal palindromic target might be 99.
            # So consider lower length (if L > 1) and higher length.
            if L > 1:
                # highest palindrome with L-1 digits: which is like all 9s.
                p_low = int("9" * (L - 1))
                if 1 <= p_low < 10**9:
                    candidates.add(p_low)
            # For a higher length, consider the smallest palindrome with L+1 digits (e.g. 1001 for L=3, but careful with constraint)
            # But only if L+1 <= 9 (because p < 10^9, maximum 9 digits)
            if L + 1 <= 9:
                # smallest palindrome with L+1 digits is 10^(L) + 1 mirrored, e.g. for 3 digits, 1001.
                # To generate that, take half length = (L+1+1)//2 = (L+2)//2, but simpler is to just use 10^(L) 
                p_high = make_palindrome("1" + "0"*((L+1)//2 - 1), L+1)
                if 1 <= p_high < 10**9:
                    candidates.add(p_high)
            
            return candidates
        
        # Our goal: choose some palindromic number y (candidate) to which the cost = sum(abs(num - y)) is minimized.
        # We know that if any real number was allowed, the optimal is the median.
        # So we compute the median of nums (if n is odd, that's the middle; if even, any number between the two mids gives same cost).
        n = len(nums)
        sorted_nums = sorted(nums)
        if n % 2 == 1:
            median = sorted_nums[n//2]
        else:
            # For even, we can just choose one of the two medians (or their integer floor/ceil).
            # We'll pick the lower median.
            median = sorted_nums[n//2 - 1]
        
        # Generate candidate palindromic numbers near median:
        candidates = gen_candidates(median)
        
        # There is possibility that the median is far from any palindrome. 
        # So we add a few additional candidates from around median by a small range.
        # We'll check a window around median (say +/- 500) if median is small.
        # But note: median can be up to 1e9. And scanning +/-500 is negligible compared to n=1e5.
        # Also note: numbers are not dense, but palindromic numbers have structure.
        # However, since our median should be near optimum, adding a window check cannot hurt.
        for offset in range(-500, 501):
            candidate = median + offset
            if candidate < 1 or candidate >= 10**9:
                continue
            s_candidate = str(candidate)
            if s_candidate == s_candidate[::-1]:  # palindrome check
                candidates.add(candidate)
        
        # Now compute cost for each candidate
        min_cost = float('inf')
        for candidate in candidates:
            cost = sum(abs(num - candidate) for num in nums)
            if cost < min_cost:
                min_cost = cost
        return min_cost