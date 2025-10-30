class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        invalid_count = 0
        
        for l in range(n):
            current_ops = 0
            current_max = nums[l]
            for r in range(l, n):
                if r > l:
                    if nums[r] < current_max:
                        current_ops += current_max - nums[r]
                    else:
                        current_max = nums[r]
                if current_ops > k:
                    invalid_count += (n - r)
                    break
        
        return total_subarrays - invalid_count