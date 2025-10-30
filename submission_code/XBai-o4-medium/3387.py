class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        m = len(nums) // 2
        if nums[m] == k:
            return 0
        elif nums[m] < k:
            res = 0
            for num in nums[m:]:
                if num < k:
                    res += k - num
                else:
                    break  # since the rest are >=k, no need to process further
            return res
        else:
            res = 0
            for num in nums[:m+1]:
                if num > k:
                    res += num - k
            return res