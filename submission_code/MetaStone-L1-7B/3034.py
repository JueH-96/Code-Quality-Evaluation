class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        if not nums:
            return 0
        sorted_nums = sorted(nums)
        merged = [sorted_nums[0]]
        for current in sorted_nums[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        total = 0
        for interval in merged:
            total += interval[1] - interval[0] + 1
        return total