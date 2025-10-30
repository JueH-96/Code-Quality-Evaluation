from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute powers of 2: pow2[k] = 2^k mod mod for k from 0 to n.
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] * 2) % mod
        
        # We need also to efficiently sum over the terms that involve a factor of 2^(j-i-1).
        # Notice: For any pair chosen with indices i < j, the contribution is:
        #   (nums[j]^2)*nums[i]*2^(j - i - 1)
        # Rearranging the summation if we fix j (the maximum element) we need:
        #   for each j>=1:
        #      nums[j]^2 * sum_{i=0}^{j-1}[ nums[i] * 2^(j-i-1) ]
        # Write 2^(j-i-1) = 2^(j-1)/2^i.
        # So the sum becomes: nums[j]^2 * 2^(j-1) * (sum_{i=0}^{j-1}[ nums[i] / 2^i ])
        # It is handy to precompute the inverse of 2 modulo mod.
        inv2 = (mod + 1) // 2

        # Precompute inverse powers: inv2pow[i] = (inv2)^i mod mod.
        inv2pow = [1] * n
        for i in range(1, n):
            inv2pow[i] = (inv2pow[i - 1] * inv2) % mod
        
        # The overall sum will be the sum for single-element groups plus 
        # the contributions from pairs and larger groups.
        # For a single element group with element a, power = a^2 * a = a^3.
        result = 0
        for a in nums:
            result = (result + a * a % mod * a) % mod

        # dp will accumulate the sum S = sum_{i=0}^{j-1} [ nums[i] * inv2^i ] as we iterate j.
        dp = 0  
        extra = 0
        # Process groups where there are at least 2 elements.
        for j in range(n):
            # For the current element nums[j] serving as maximum,
            # if j>0 then there exists at least one smaller element to serve as minimum.
            if j > 0:
                # Contribution from groups with maximum index j is:
                #   nums[j]^2 * 2^(j-1) * (sum_{i=0}^{j-1} [nums[i] / 2^i])
                contrib = (nums[j] * nums[j]) % mod
                contrib = (contrib * pow2[j - 1]) % mod
                contrib = (contrib * dp) % mod
                extra = (extra + contrib) % mod
            # Update dp to include the current element at index j
            dp = (dp + nums[j] * inv2pow[j]) % mod
        
        result = (result + extra) % mod
        return result