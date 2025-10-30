class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changed = True
        while changed:
            changed = False
            for i in range(n - 1):
                if nums[i] > nums[i+1]:
                    original_i = nums[i]
                    original_j = nums[i+1]
                    nums[i] = original_i & original_j
                    nums[i+1] = original_i | original_j
                    if nums[i] != original_i or nums[i+1] != original_j:
                        changed = True
                        
        nums.sort(reverse=True)
        max_sum_sq = 0
        mod = 10**9 + 7
        for i in range(k):
            max_sum_sq = (max_sum_sq + (nums[i] * nums[i]) % mod) % mod
            
        return max_sum_sq