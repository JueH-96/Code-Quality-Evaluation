class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = set()
        current = 1
        total_sum = 0
        
        while len(nums) < n:
            if (target - current) not in nums:
                nums.add(current)
                total_sum = (total_sum + current) % MOD
            current += 1
        
        return total_sum