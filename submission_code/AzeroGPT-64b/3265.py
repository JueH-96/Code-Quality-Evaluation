class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        first = defaultdict(int)
        first[nums[0]] = 0 
        suffix_sum = [0] * len(nums)
        suffix_sum[-1] = nums[-1]
        for i in reversed(range(len(nums) - 1)):
            suffix_sum[i] = nums[i] + suffix_sum[i + 1]
    
        ans = 0 - (1 << 60) if k > 0 else 0
        for i, num in enumerate(nums):
            if num + k in first:
                ans = max(ans, suffix_sum[first[num + k]] - (0 if first[num + k] else 0) + num)
            if num - k in first:
                ans = max(ans, suffix_sum[first[num - k]] - (0 if first[num - k] else 0) + num)
            first[num] = i + 1
        return ans