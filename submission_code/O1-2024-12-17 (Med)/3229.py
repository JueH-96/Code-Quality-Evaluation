class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        We want all elements of nums to be changed into the same palindromic integer p < 10^9,
        minimizing the total cost of sum(|nums[i] - p|).

        Observing that for sum of absolute differences, the best unconstrained target is around
        the median(s). Constraining to palindromic targets suggests we only need to consider
        palindromes "close" to the median(s). We'll do the following:

         1. Find the median(s) of nums:
            - If n is odd, there's one median m.
            - If n is even, there are two medians m1, m2.
         2. For each median, generate a small set of "nearby" palindromes (including possibly
            palindromes of length len(m)-1, len(m), len(m)+1). We'll do this by:
            - Converting m to string,
            - Creating palindromes by mirroring the left half (and considering +/- 1 in that half),
            - Also repeating this logic for d-1, d, d+1 digit lengths.
            - We keep only values in [1, 10^9 - 1].
         3. Compute sum(|nums[i] - candidate|) for each candidate palindrome; the answer is
            the minimum cost over all candidates generated from the median(s).
        """

        import sys
        sys.setrecursionlimit(10**7)

        # A helper function to build an integer palindrome from a 'half' string.
        # If odd_len is True, we mirror all but the last digit of 'half_str'.
        # Example: half_str="123", odd_len=True -> "12321"; odd_len=False -> "123321"
        def make_palindrome(half_str: str, odd_len: bool) -> int:
            if odd_len:
                # "abc" -> "abcba" (mirror everything but the last char of half_str)
                return int(half_str + half_str[-2::-1])  # skip the last char on reversing
            else:
                # "abc" -> "abccba"
                return int(half_str + half_str[::-1])

        # Given an integer x, generate a set of palindromes 'close' to x
        # by mirroring the first half, and also incrementing/decrementing the half.
        # We do this for length len(x), len(x)-1, and len(x)+1 to catch edge cases
        # like going from 999 -> 1001, etc.
        def get_close_palindromes(x: int) -> set:
            s = str(x)
            length = len(s)
            candidates = set()

            # We'll define a helper to generate palindromes of a given length d
            # from a "half" string (depending on whether d is odd or even).
            def generate_from_half(half: str, d: int):
                # If the half is too short or too long, skip
                if len(half) < (d + 1) // 2:
                    return
                # Trim or keep exactly the needed left half
                needed = (d + 1) // 2
                half_str = half[:needed]
                # Build palindrome
                p = make_palindrome(half_str, odd_len=(d % 2 == 1))
                if 1 <= p < 10**9:
                    candidates.add(p)

            # Try lengths in [length-1, length, length+1] (all >=1)
            for d in range(max(1, length - 1), length + 2):
                # The "central half-length" needed
                half_len = (d + 1) // 2

                # We'll construct half from the first half_len digits of x (if length >= half_len)
                # or if length < half_len, we might have to pad. But realistically if d < length,
                # the "half" from x might not match. We'll still do best we can.

                # 1) Direct half
                if length >= half_len:
                    direct_half = s[:half_len]
                else:
                    # If needed half is bigger than s, just use s (leading to smaller number)
                    direct_half = s

                # 2) We'll try direct_half, int(direct_half)+1, int(direct_half)-1 (if possible)
                #    to get "nearby" halves.
                for diff in [0, 1, -1]:
                    try:
                        half_int = int(direct_half) + diff
                    except ValueError:
                        continue
                    if half_int < 0:
                        continue
                    new_half_str = str(half_int)
                    generate_from_half(new_half_str, d)

            return candidates

            # We'll gather all these possible palindromes in 'candidates'.

        # 1. Find median(s)
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        if n % 2 == 1:
            medians = [nums_sorted[n // 2]]
        else:
            medians = [nums_sorted[n // 2 - 1], nums_sorted[n // 2]]

        # 2. For each median, gather candidate palindromes
        pal_candidates = set()
        for m in medians:
            # Clip 'm' to [1, 10^9 - 1] for consistency
            if m < 1:
                m = 1
            if m >= 10**9:
                m = 10**9 - 1
            pal_candidates.update(get_close_palindromes(m))

        # Also, in case medians differ, we might want to carefully look at both,
        # but we already do that by iterating over medians.

        # 3. Compute cost for each candidate and take the minimum
        best_cost = float('inf')
        prefix_sum = None  # We'll just do a direct sum of differences for each candidate.

        for pal in pal_candidates:
            cost = 0
            for val in nums:
                cost += abs(val - pal)
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost = cost

        return best_cost