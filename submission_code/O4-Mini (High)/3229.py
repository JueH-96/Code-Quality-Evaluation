from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Helper to build a palindrome from a prefix string for length L
        def make_pal(prefix: str, L: int) -> int:
            if L % 2 == 0:
                rev = prefix[::-1]
            else:
                rev = prefix[:-1][::-1]
            return int(prefix + rev)
        
        # Largest palindromic integer <= y
        def floor_pal(y: int) -> int:
            s = str(y)
            L = len(s)
            half = (L + 1) // 2
            prefix = s[:half]
            pal0 = make_pal(prefix, L)
            if pal0 <= y:
                return pal0
            # Need to decrement the prefix
            pnum = int(prefix) - 1
            # If prefix underflows or loses a digit, fallback to all 9's of length L-1
            if pnum <= 0 or len(str(pnum)) < half:
                if L == 1:
                    # single digit case won't actually happen here because
                    # any single digit is already palindrome <= y
                    return y
                return int('9' * (L - 1))
            # Otherwise mirror the decremented prefix
            pstr = str(pnum).zfill(half)
            return make_pal(pstr, L)
        
        # Smallest palindromic integer >= y
        def ceil_pal(y: int) -> int:
            s = str(y)
            L = len(s)
            half = (L + 1) // 2
            prefix = s[:half]
            pal0 = make_pal(prefix, L)
            if pal0 >= y:
                return pal0
            # Need to increment the prefix
            pnum = int(prefix) + 1
            # If prefix overflows (gains a digit), we must jump to 1 0...0 1 of length L+1
            if len(str(pnum)) > half:
                # e.g. for L=4 => "100001", for L=5 => "1000001"
                return int('1' + '0' * (L - 1) + '1')
            # Otherwise mirror the incremented prefix
            pstr = str(pnum).zfill(half)
            return make_pal(pstr, L)
        
        n = len(nums)
        # Sort a copy to find median(s)
        arr = sorted(nums)
        candidates = set()
        if n % 2 == 1:
            # Odd length: unique median
            m = arr[n // 2]
            candidates.add(floor_pal(m))
            candidates.add(ceil_pal(m))
        else:
            # Even length: any y in [l, r] is optimal unconstrained
            l = arr[n//2 - 1]
            r = arr[n//2]
            # Check if there's a palindrome inside [l, r]
            pmid = ceil_pal(l)
            if pmid <= r:
                candidates.add(pmid)
            else:
                # No palindrome inside the median interval
                candidates.add(floor_pal(l))
                candidates.add(ceil_pal(r))
        # Filter out palindromes >= 10^9 (not allowed)
        LIMIT = 10**9
        candidates = [y for y in candidates if 1 <= y < LIMIT]
        
        # Compute the cost for each candidate and pick the minimum
        best = 10**30
        for y in candidates:
            cost = 0
            for x in nums:
                cost += abs(x - y)
            if cost < best:
                best = cost
        return best