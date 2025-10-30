class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = []
        current = 1
        
        while len(nums) < n:
            # Check if current number can be added to the nums array
            if target - current not in nums:
                nums.append(current)
            current += 1
        
        return sum(nums) % MOD