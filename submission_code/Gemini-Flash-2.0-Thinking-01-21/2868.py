class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                is_continuous = True
                if len(subarray) > 0:
                    min_val = min(subarray)
                    max_val = max(subarray)
                    if max_val - min_val > 2:
                        is_continuous = False
                if is_continuous:
                    count += 1
        return count