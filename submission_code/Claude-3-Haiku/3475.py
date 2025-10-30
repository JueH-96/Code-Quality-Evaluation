class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        
        count_0 = nums.count(0)
        if count_0 == 0:
            return 0
        
        if count_0 == n:
            return (n + 2) // 3
        
        min_ops = float('inf')
        for i in range(n - 2):
            if nums[i] == 0:
                ops = 1
                for j in range(i, i + 3):
                    if nums[j] == 0:
                        ops += 1
                min_ops = min(min_ops, ops)
        
        return min_ops if min_ops != float('inf') else -1