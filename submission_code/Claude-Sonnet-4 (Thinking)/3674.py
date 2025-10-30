class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            operations = 0
            max_so_far = nums[i]
            for j in range(i, n):
                max_so_far = max(max_so_far, nums[j])
                operations += max_so_far - nums[j]
                if operations <= k:
                    count += 1
                else:
                    break
        
        return count