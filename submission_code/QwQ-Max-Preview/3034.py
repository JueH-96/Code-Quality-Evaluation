class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Sort the intervals based on their start points
        nums.sort(key=lambda x: x[0])
        if not nums:
            return 0
        
        merged = [nums[0]]
        for interval in nums[1:]:
            last_start, last_end = merged[-1]
            current_start, current_end = interval
            if current_start <= last_end:
                # Merge the intervals by updating the end of the last interval
                merged[-1][1] = max(last_end, current_end)
            else:
                merged.append(interval)
        
        # Calculate the total number of covered points
        total = 0
        for start, end in merged:
            total += end - start + 1
        return total