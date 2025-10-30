class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        We want to make all elements of nums equal to the same palindromic integer y (< 10^9)
        with minimal total cost, where cost is sum of |nums[i] - y|.

        Observing that sum of absolute differences is smallest near the median(s), we only need
        to consider palindromic candidates between [min(nums), max(nums)] (clamped to [1, 999999999]).

        Approach:
          1) Generate all palindromes p < 10^9. (There are fewer than 10^5 of these in total.)
          2) Sort and prefix-sum nums for O(log n) cost-of-change computation.
          3) Restrict candidates to palindromes in [low, high], where
               low = max(1, min(nums)), high = min(999999999, max(nums)).
          4) For each candidate palindrome p, compute cost = sum(|nums[i] - p|).
             Use prefix sums + binary search to do this in O(log n).
          5) Return the minimal such cost.

        This is efficient since we'll at most evaluate ~10^5 palindromes, each in O(log n),
        which is borderline but typically feasible with careful implementation in Python.
        """

        import bisect
        
        # 1) Generate all palindromes < 10^9
        palindromes = []
        
        # A helper to mirror a left half to form palindromes
        def make_pal(num_str, odd_digits):
            """
            Given num_str as the left part (for odd-digit palindrome it includes the middle),
            and a flag odd_digits, return both the palindrome as integer
            and, if odd_digits is True, the duplicate formed by removing the middle digit mirror
            (that forms an even-digit palindrome). We'll handle them carefully to avoid duplicates.
            """
            # For example, left="123", odd_digits -> palindrome "123321"
            # If odd_digits is True, we skip the last char in left when mirroring
            if odd_digits:
                return int(num_str + num_str[-2::-1])
            else:
                return int(num_str + num_str[::-1])
        
        # We generate palindromes by constructing the first half (and middle for odd length).
        # We handle digit lengths 1..9. For a d-digit palindrome:
        #   - if d is odd = 2k+1, we pick the left half as k+1 digits (first digit != '0')
        #   - if d is even = 2k, we pick the left half as k digits (first digit != '0')
        
        # Maximum 9-digit palindromes
        # We'll systematically build them and store if <10^9.
        for digits in range(1, 10):  # 1..9 digits
            half_len = (digits + 1) // 2  # length of "left side" for building
            start = 10**(half_len - 1)  # e.g., for half_len=2, start=10, for half_len=1, start=1
            end = 10**(half_len)        # e.g., for half_len=2, end=100
            
            for half in range(start, end):
                s = str(half)
                if digits % 2 == 0:
                    # even-digit palindrome
                    p = make_pal(s, odd_digits=False)
                else:
                    # odd-digit palindrome
                    p = make_pal(s, odd_digits=True)
                
                if p < 10**9:
                    palindromes.append(p)
                
                # For odd-digit lengths, also consider the even-digit palindrome made
                # by mirroring the entire half (this is effectively next smaller digit length 
                # if we did not do it separately, but let's skip duplication for even-case).
                # Actually we handle even digits in a separate branch. So we skip it here
                # to avoid duplicates.
        
        # Remove duplicates if any (shouldn't be many if any) and sort
        palindromes = list(set(palindromes))
        palindromes.sort()
        
        # 2) Sort nums and build prefix sums
        nums_sorted = sorted(nums)
        n = len(nums)
        prefix_sum = [0]*(n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums_sorted[i]
        
        # A helper to compute sum of |nums[i]-x| in O(log n) using prefix sums
        def total_cost(x):
            # position pos = number of elements <= x
            pos = bisect.bisect_right(nums_sorted, x)
            # sum_left = prefix_sum of first pos elements
            sum_left = prefix_sum[pos]
            # count_left = pos
            # sum_right = prefix_sum(n) - sum_left
            sum_right = prefix_sum[n] - sum_left
            count_left = pos
            count_right = n - pos
            
            # cost = x*count_left - sum_left + sum_right - x*count_right
            return x*count_left - sum_left + (sum_right - x*count_right)
        
        # 3) Restrict palindrome candidates in [low, high]
        low = max(1, min(nums))
        high = min(999999999, max(nums))
        
        # Find subrange in 'palindromes' that lies within [low, high]
        left_idx = bisect.bisect_left(palindromes, low)
        right_idx = bisect.bisect_right(palindromes, high)
        
        # We'll collect candidate palindromes from left_idx..right_idx
        # plus we check boundary neighbors if out-of-range is needed.
        candidates = []
        if left_idx > 0:
            candidates.append(palindromes[left_idx-1])
        candidates.extend(palindromes[left_idx:right_idx])
        if right_idx < len(palindromes):
            candidates.append(palindromes[right_idx])
        
        # Filter out those not within [1, 10^9), but we do keep them if they're boundary neighbors
        # for possible minimal cost just outside [low, high].
        final_candidates = []
        for c in candidates:
            if 1 <= c < 10**9:
                final_candidates.append(c)
        
        # 4) Compute minimal cost among these candidates
        ans = float('inf')
        for pal in final_candidates:
            ans = min(ans, total_cost(pal))
        
        return ans