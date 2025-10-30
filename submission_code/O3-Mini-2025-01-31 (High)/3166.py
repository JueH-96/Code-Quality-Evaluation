class Solution:
    def minGroupsForValidAssignment(self, nums: "List[int]") -> int:
        from collections import Counter
        n = len(nums)
        # Count the frequency for each distinct number.
        freq = Counter(nums)
        m = len(freq)  # number of distinct values
        # For efficiency, group frequencies by their value.
        freqCount = Counter(freq.values())
        # Prepare a list of (frequency, count) pairs.
        freqPairs = list(freqCount.items())
        
        # Our goal is to find the minimal k (with m <= k <= n) such that a valid assignment exists.
        # If we use k groups then each group is of size either a or a+1 where a = n // k.
        # Also, if a = n//k then k must lie in [ floor(n/(a+1)) + 1,  floor(n/a) ].
        # For each distinct value with frequency f, if we cut it into p groups (all parts being a or a+1)
        # then we must have:   p*a <= f <= p*(a+1).
        # Equivalently, p must be an integer in [ L, U ] where:
        #       L = ceil(f/(a+1)) = (f + a) // (a+1)   and   U = floor(f/a).
        # A necessary and sufficient condition for a valid assignment is that
        #   Σₓ Lₓ  ≤  k  ≤  Σₓ Uₓ.
        #
        # Notice that once k is chosen, a is determined as n//k.
        # Conversely, if we “guess” a candidate a (which equals n//k) then k must be in:
        #      [k_low, k_high] where k_low = floor(n/(a+1)) + 1 and k_high = floor(n/a).
        # Also, k must satisfy:   sum(Lₓ) ≤ k ≤ sum(Uₓ), and naturally k ≥ m.
        #
        # We now iterate on candidate values of a.
        # Note that when k is as small as m, a_max = n // m.
        ans = 10**9
        max_a = n // m  # a = n//k can be at most n//m
        for a in range(1, max_a + 1):
            # k must be in the interval:
            k_low = (n // (a + 1)) + 1   # smallest k with n//k == a
            k_high = n // a              # largest k with n//k == a
            
            # Now sum over the distinct frequencies:
            sum_L = 0  # total minimum groups needed from all distinct numbers
            sum_U = 0  # total maximum groups available from all distinct numbers
            possible = True
            for f, cnt in freqPairs:
                L_f = (f + a) // (a + 1)  # ceil( f/(a+1) )
                U_f = f // a              # floor(f/a)
                if L_f > U_f:
                    possible = False
                    break
                sum_L += L_f * cnt
                sum_U += U_f * cnt
            if not possible:
                continue
            
            # k must also be at least m.
            lower_bound = max(k_low, sum_L, m)
            upper_bound = min(k_high, sum_U)
            if lower_bound <= upper_bound:
                # Choose the smallest candidate k in the intersection.
                if lower_bound < ans:
                    ans = lower_bound
        return ans

# For testing locally:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minGroupsForValidAssignment([3,2,3,2,3]))  # Expected output: 2 
    # Example 2:
    print(sol.minGroupsForValidAssignment([10,10,10,3,1,1]))  # Expected output: 4
    # Additional tests:
    print(sol.minGroupsForValidAssignment([1,2]))         # Expected output: 2
    print(sol.minGroupsForValidAssignment([1,1,2]))       # Expected output: 2
    print(sol.minGroupsForValidAssignment([1,1,1,2,2,3])) # Expected output: 4