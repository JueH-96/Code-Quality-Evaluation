class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        total = 0
        dp = 0  # Weighted sum of previous minimums
        
        for num in nums:
            # When num is the maximum:
            # - dp contains weighted sum of all possible minimums from previous elements
            # - num itself can be both min and max (single element subset)
            contribution = num * num % MOD * (dp + num) % MOD
            total = (total + contribution) % MOD
            
            # Update dp for next iteration
            # Each previous minimum's weight doubles (one more element between)
            # Plus num itself with weight 1
            dp = (2 * dp + num) % MOD
        
        return total