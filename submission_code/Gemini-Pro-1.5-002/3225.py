class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        max_length = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                counts = {}
                good = True
                for num in subarray:
                    counts[num] = counts.get(num, 0) + 1
                    if counts[num] > k:
                        good = False
                        break
                if good:
                    max_length = max(max_length, len(subarray))
        return max_length