class Solution:
    def maxScore(self, nums: List[int]) -> int:
        from math import gcd
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(nums)
        if n == 0:
            return 0  # Edge case: empty array
        prefix_GCD = [0] * n
        prefix_LCM = [0] * n
        suffix_GCD = [0] * n
        suffix_LCM = [0] * n
        
        prefix_GCD[0] = nums[0]
        prefix_LCM[0] = nums[0]
        for i in range(1, n):
            prefix_GCD[i] = gcd(prefix_GCD[i - 1], nums[i])
            prefix_LCM[i] = lcm(prefix_LCM[i - 1], nums[i])
        
        suffix_GCD[n - 1] = nums[n - 1]
        suffix_LCM[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_GCD[i] = gcd(nums[i], suffix_GCD[i + 1])
            suffix_LCM[i] = lcm(nums[i], suffix_LCM[i + 1])
        
        # Compute factor score without removing any element
        total_GCD = prefix_GCD[n - 1]
        total_LCM = prefix_LCM[n - 1]
        max_factor_score = total_GCD * total_LCM
        
        # Check all possibilities by removing at most one element
        for i in range(n):
            if n == 1:
                # Removing the only element results in an empty array
                factor_score = 0
            else:
                if i == 0:
                    GCD_without_i = suffix_GCD[1]
                    LCM_without_i = suffix_LCM[1]
                elif i == n - 1:
                    GCD_without_i = prefix_GCD[n - 2]
                    LCM_without_i = prefix_LCM[n - 2]
                else:
                    GCD_without_i = gcd(prefix_GCD[i - 1], suffix_GCD[i + 1])
                    LCM_without_i = lcm(prefix_LCM[i - 1], suffix_LCM[i + 1])
                factor_score = GCD_without_i * LCM_without_i
            max_factor_score = max(max_factor_score, factor_score)
        
        return max_factor_score