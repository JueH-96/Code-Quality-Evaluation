class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        We want to choose a single palindromic number y (with 1 <= y < 10^9)
        that minimizes the total cost = sum( |nums[i] - y| ).

        Observing that the sum of absolute differences from a set of numbers
        is generally minimized by picking the median, we can focus on palindromes
        "near" the median(s) of the array. We then compute the cost for each
        candidate palindrome and return the minimum.

        Steps:
        1) Sort nums. Identify up to two "median" values:
           - If n is odd, median is nums[n // 2].
           - If n is even, there are two medians: nums[n//2 - 1] and nums[n//2].
        2) For each median candidate m, generate a small set of palindromes around m.
           We do this by:
             - Constructing the palindrome from m directly (mirror digits).
             - Slightly incrementing/decrementing the "middle part" to handle
               the next or previous palindromes.
             - Considering possibly one fewer digit (e.g. if m ~ 1000, check 999)
               and one more digit (e.g. if m ~ 999, check 1001), provided they are
               < 10^9 and > 0.
        3) For each candidate palindrome p, compute the total cost = sum(|x - p|).
        4) Return the minimum of these costs.

        This approach takes O(n) time to compute the costs for each candidate,
        but we only generate a handful of palindromes per median. Thus overall
        time complexity is O(n), which is acceptable for n up to 1e5.
        """

        import bisect
        
        # -------------------------
        # Helper Functions
        # -------------------------
        
        def is_palindrome(x: int) -> bool:
            s = str(x)
            return s == s[::-1]
        
        def make_palindrome_half(s: str, mirror_middle: bool) -> int:
            """
            Given a string s representing digits, produce its palindrome by mirroring.
            If mirror_middle = True and len(s) is odd, we keep the middle digit as is,
            mirroring only the first half around it.
            If mirror_middle = False, we mirror the exact first half into the second half,
            ignoring "middle" logic. (Used in even-digit context or forcibly adjusting.)
            
            Example: s='123', mirror_middle=True -> palindrome='12321'
                     s='123', mirror_middle=False -> palindrome='123321' (not typical for the same length).
            We'll mostly rely on the length-based approach: for length L, if L is odd, we mirror around the middle digit.
            """
            L = len(s)
            if L == 1:
                return int(s)  # Single-digit is already a palindrome
            
            if mirror_middle and (L % 2 == 1):
                mid = L // 2
                left = s[:mid]    # everything before middle
                middle = s[mid]   # middle char
                # the 'right' is determined by left reversed
                p_str = left + middle + left[::-1]
            else:
                # mirrored exactly half ignoring a 'middle'
                half_len = (L + 1) // 2  # e.g. for L=4, half_len=2; L=3, half_len=2
                left = s[:half_len]
                p_str = left + left[:-1 if L%2==1 else None][::-1]
            return int(p_str)
        
        def get_nearby_palindromes(x: int) -> List[int]:
            """
            Return a set of palindromes close to x, including possible edge cases
            from adjusting the 'middle' digits, also from shorter or longer digit counts.
            We'll limit ourselves to palindromes < 10^9 and > 0.
            """
            s = str(x)
            length = len(s)
            candidates = set()
            
            # Generate the "base" palindrome from s by mirroring
            base_pal = make_palindrome_half(s, mirror_middle=True)
            candidates.add(base_pal)
            
            # A small helper to add or subtract 1 from the 'first half' (including middle if odd).
            # Then mirror again and see if it's valid.
            def adjust_and_mirror(s_digits: str, delta: int):
                if not s_digits:
                    return
                # convert to int, add delta, convert back
                val = int(s_digits)
                val += delta
                # if going negative or too large, skip
                if val < 0:
                    return
                new_s = str(val)
                # ensure we only produce equal or smaller length if possible
                if len(new_s) > len(s_digits):
                    return
                # now we mirror it properly to get a candidate
                # We handle the final length to match original s_digits length:
                # fill with leading zeros if needed. For instance, if s_digits='100'
                # and delta=-1 => 99 => new_s='99', we can then produce a palindrome of
                # length 2 or 3. We'll consider both ways carefully.
                
                # We'll try to produce a palindrome with the same digit-length as 's_digits'
                # if the new_s is shorter, we can left-pad zeros, but normally that wouldn't
                # produce a standard decimal. So let's just produce the palindrome from new_s
                # and consider if length changed.
                candidate = make_palindrome_half(new_s, mirror_middle=(len(s_digits) % 2 == 1))
                return candidate
            
            # We attempt to adjust the half (including the middle digit if odd)
            half_len = (length + 1) // 2
            half_str = s[:half_len]  # the portion we mirror
            for d in [-1, +1]:
                adj = adjust_and_mirror(half_str, d)
                if adj is not None:
                    candidates.add(adj)
            
            # For palindrome with shorter length e.g. if x=1000 -> consider a 3-digit palindrome (999)
            # but only if length > 1
            if length > 1:
                # largest (length-1)-digit palindrome is something like 999.. (length-1 times)
                shorter_pal = int('9' * (length-1))
                candidates.add(shorter_pal)
            
            # For palindrome with one more digit e.g. x=999 -> consider 1001
            # If length < 9, we can try 10..01 form, e.g. for length=3 => 4-digit '1001'
            if length < 9:
                longer_pal = int('1' + ('0' * (length-1)) + '1')
                candidates.add(longer_pal)
            
            # Also consider the direct "mirror" if we forced an even-digit approach
            # or forced an odd-digit approach. This can differ if length is odd:
            # Example: x=123 => mirror_middle=True gives '12321'; mirror_middle=False might try '123321'
            # We add both in case there's a digit mismatch.
            forced_even = make_palindrome_half(s, mirror_middle=False)
            candidates.add(forced_even)
            
            # Clean up invalid or out-of-range results
            valid_cands = []
            for c in candidates:
                if c > 0 and c < 10**9:
                    valid_cands.append(c)
            
            return valid_cands
        
        # -------------------------
        # Main Logic
        # -------------------------
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        # Potential median candidates
        if n % 2 == 1:
            medians = [nums_sorted[n // 2]]
        else:
            medians = [nums_sorted[n // 2 - 1], nums_sorted[n // 2]]
        
        # We gather palindromes near these medians
        pal_candidates = set()
        for m in medians:
            pal_candidates.update(get_nearby_palindromes(m))
        
        # We also might include near the min and max of the array, in case
        # the best palindrome lies near extremes
        # (This can sometimes help if the array is mostly large or mostly small.)
        min_val, max_val = nums_sorted[0], nums_sorted[-1]
        pal_candidates.update(get_nearby_palindromes(min_val))
        pal_candidates.update(get_nearby_palindromes(max_val))
        
        # We'll compute the sum of absolute differences for these palindrome candidates
        # and return the minimum.
        
        def total_cost(p):
            # sum of |nums[i] - p|
            # We'll do this in one pass (O(n)).
            # For large n, we want to do it once per candidate (which is a small set).
            s = 0
            for val in nums:
                diff = val - p
                if diff < 0:
                    diff = -diff
                s += diff
            return s
        
        ans = float('inf')
        for pal in pal_candidates:
            cost = total_cost(pal)
            if cost < ans:
                ans = cost
        
        return ans