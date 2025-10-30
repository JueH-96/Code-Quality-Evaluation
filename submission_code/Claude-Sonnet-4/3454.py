class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        operations = 0
        running_balance = 0
        
        for i in range(n):
            diff = target[i] - nums[i]
            
            # If we need more positive change than what we have
            if diff > running_balance:
                operations += diff - running_balance
            # If we need more negative change than what we have
            elif diff < running_balance:
                operations += running_balance - diff
            
            # Update running balance to the current difference
            running_balance = diff
        
        return operations