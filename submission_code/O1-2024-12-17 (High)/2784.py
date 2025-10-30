class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        
        # f(i) will represent the sum of the minimums of all non-empty subsets
        # formed from the sorted array slice nums[:i+1].
        #
        # It satisfies the recurrence relation:
        #   f(0) = nums[0]
        #   f(i) = 2 * f(i - 1) + nums[i]
        #
        # For each element nums[i], the number of subsets having nums[i] as the maximum
        # is all subsets (including possibly empty) drawn from nums[:i], combined with nums[i].
        # If T is a subset from nums[:i], then min(T ∪ { nums[i] }) = min(T)
        # (except when T is empty, then the min is nums[i]).
        #
        # Therefore, the sum of minimums over all subsets whose max is nums[i] is:
        #   f(i - 1) + nums[i]
        #
        # The power contributed by these subsets is then:
        #   (nums[i]^2) * (f(i - 1) + nums[i])
        #
        # We accumulate this for all i.
        
        f_prev = 0  # f(-1) considered as 0
        ans = 0
        
        for x in nums:
            # Contribution from subsets whose max is x:
            #   x^2 * (f_prev + x)
            contrib = (x * x) % MOD
            contrib = (contrib * ((f_prev + x) % MOD)) % MOD
            ans = (ans + contrib) % MOD
            
            # Update f(i) = 2 * f(i-1) + nums[i]
            f_prev = ((2 * f_prev) % MOD + x) % MOD
        
        return ans