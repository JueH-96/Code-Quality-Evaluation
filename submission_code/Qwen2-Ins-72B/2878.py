class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        for i in range(n):
            if nums[i] > 0:
                if i + k > n or prefix_sum[i + k] - prefix_sum[i] < nums[i] * k:
                    return False
                prefix_sum[i + k] -= nums[i] * k
                prefix_sum[i + 1] += nums[i] * k
                nums[i] = 0
        
        return all(num == 0 for num in nums)