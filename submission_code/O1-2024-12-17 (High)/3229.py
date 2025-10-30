class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        import bisect

        # Sort the array for efficient cost computation via prefix sums
        s = sorted(nums)
        n = len(s)

        # Build prefix sums so we can compute sum of absolute differences in O(log n)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]

        def cost_for_p(p):
            """
            Compute sum of |nums[i] - p| over all i using prefix sums.
            idx = number of elements <= p
            cost = (p*idx - sum_of_those_idx_elements) + ((sum_of_rest) - p*(n - idx))
            """
            idx = bisect.bisect_right(s, p)  # number of elements <= p
            left_cost = p * idx - prefix[idx]
            right_cost = (prefix[n] - prefix[idx]) - p * (n - idx)
            return left_cost + right_cost

        def generate_palindrome(half_int, length):
            """
            Given an integer half_int (the 'left' part) and the total length of the palindrome,
            build the corresponding palindrome (if possible).
            Return None if it cannot be made into a valid palindrome of the given length.
            """
            half_str = str(half_int)
            # The left half must have at most (length+1)//2 digits
            half_len = (length + 1) // 2
            if len(half_str) > half_len:
                return None
            # Pad with leading zeros if necessary (e.g., when we do half-1 and it shrinks)
            half_str = half_str.zfill(half_len)

            # Construct the full palindrome string
            if length % 2 == 0:
                # even length, mirror entire half_str
                pal_str = half_str + half_str[::-1]
            else:
                # odd length, do not repeat the middle digit
                pal_str = half_str + half_str[-2::-1]

            # Disallow leading '0' (that would be fewer digits)
            if pal_str[0] == '0':
                return None

            return int(pal_str)

        def get_candidates(num_str):
            """
            Generate a set of palindromic candidates near the integer represented by num_str.
            We'll consider:
              1. Palindromes formed by mirroring (left-1), left, (left+1)
              2. The "edge_up" = 10^length + 1 (if it is a palindrome and < 10^9)
              3. The "edge_down" = 10^(length-1) - 1 (if palindrome and >= 1)
            """
            length = len(num_str)
            left_part = int(num_str[:(length + 1) // 2])  # the left half as integer

            candidates = set()
            # Check three palindromes from left-1, left, left+1
            for half in [left_part - 1, left_part, left_part + 1]:
                if half >= 0:
                    p = generate_palindrome(half, length)
                    if p is not None:
                        candidates.add(p)

            # edge_up = 10^length + 1
            edge_up = 10 ** length + 1
            if edge_up < 10**9:
                # check if it is a palindrome
                su = str(edge_up)
                if su == su[::-1]:
                    candidates.add(edge_up)

            # edge_down = 10^(length-1) - 1 (only makes sense if length > 1)
            if length > 1:
                edge_down = 10 ** (length - 1) - 1
                if edge_down >= 1:
                    sd = str(edge_down)
                    if sd == sd[::-1]:
                        candidates.add(edge_down)

            return candidates

        # Find medians (for odd n there's one median, for even n there are two)
        # Any integer in [s[(n-1)//2], s[n//2]] could be the "best" unconstrained, so we check both.
        med1 = s[(n - 1) // 2]
        med2 = s[n // 2]

        # Clamp them to at most 999999999, since the target palindrome must be < 10^9
        def clamp(x):
            return min(x, 10**9 - 1)

        M1 = clamp(med1)
        M2 = clamp(med2)

        # Collect candidate palindromes from both medians
        candidates = set()
        candidates.update(get_candidates(str(M1)))
        candidates.update(get_candidates(str(M2)))

        # Always consider 1 and 999999999 (both are valid palindromes)
        candidates.add(1)
        candidates.add(999999999)

        # Filter out invalid ranges
        # (must be >= 1 and < 10^9)
        final_candidates = [p for p in candidates if 1 <= p < 10**9]

        # Compute the minimal cost among these candidates
        answer = min(cost_for_p(p) for p in final_candidates)
        return answer