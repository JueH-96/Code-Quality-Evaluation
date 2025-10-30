class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 1000000007
        nums.sort()
        total = 0
        prev_sum = 0
        
        for i in range(len(nums)):
            # For each number as max, it will be paired with all possible combinations of smaller numbers
            # prev_sum keeps track of sum of all possible minimums for previous numbers
            total = (total + (nums[i] * nums[i] % MOD) * nums[i] % MOD) % MOD  # Single element group
            
            if i > 0:
                # For groups with more than one element
                # nums[i] will be max, and we need to multiply with all possible minimums
                total = (total + (nums[i] * nums[i] % MOD) * prev_sum % MOD) % MOD
            
            # Update prev_sum for next iteration
            # Each previous number can be minimum in combination with current number
            prev_sum = (prev_sum * 2 + nums[i]) % MOD
            
        return total