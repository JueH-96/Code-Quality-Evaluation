class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # First sort the intervals based on the start value.
        nums.sort(key=lambda x: x[0])
        
        merged = []
        for interval in nums:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        total_points = 0
        for start, end in merged:
            total_points += end - start + 1
        
        return total_points