class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # For each position i, we calculate contribution when nums[i] is minimum
        total = 0
        
        # We'll maintain a running sum of nums[j]^2 * 2^(j-i-1) for all j >= i
        # As we move i from right to left, we update this sum
        suffix_sum = 0
        
        for i in range(n-1, -1, -1):
            # Add contribution when nums[i] is both min and max (single element subset)
            total = (total + nums[i] * nums[i] * nums[i]) % MOD
            
            # Add contribution when nums[i] is min and some j > i is max
            total = (total + nums[i] * suffix_sum) % MOD
            
            # Update suffix_sum for next iteration
            # When we move from i to i-1, all previous terms get multiplied by 2
            # and we add the new term nums[i]^2
            suffix_sum = (suffix_sum * 2 + nums[i] * nums[i]) % MOD
        
        return total