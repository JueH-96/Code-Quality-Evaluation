class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        res, total = 1, nums[numOperations] - nums[0]
        for i in range(len(nums) - (numOperations + 1)):
            total_converter = nums[i + numOperations + 1] - nums[i]
            manipulations, remainders, remainders_possible = divmod(total_converter - total, k), [0, 0], [0, 0]
            for remainder in remainders:
                manipulations_new = manipulations + remainder
                if manipulations_new <= numOperations:
                    res, total = max(res, (numOperations + i + 1) - manipulations_new + 1), total_converter
                remainders_possible[0] += nums[i + numOperations + 1] - nums[i + numOperations] - k
                remainders_possible[1] += nums[i + numOperations] - nums[i] + k
            remainders = [i % k for i in sorted(remainders_possible)]
        return res