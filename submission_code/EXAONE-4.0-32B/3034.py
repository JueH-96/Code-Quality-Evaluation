class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        if not nums:
            return 0
        nums.sort()
        merged = []
        start, end = nums[0]
        for i in range(1, len(nums)):
            s, e = nums[i]
            if s <= end:
                end = max(end, e)
            else:
                merged.append([start, end])
                start, end = s, e
        merged.append([start, end])
        
        total_points = 0
        for inter in merged:
            total_points += (inter[1] - inter[0] + 1)
        return total_points