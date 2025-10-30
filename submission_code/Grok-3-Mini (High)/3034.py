class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Sort the intervals based on the start time
        nums.sort(key=lambda x: x[0])
        
        # Initialize the current start and end with the first interval
        current_start = nums[0][0]
        current_end = nums[0][1]
        
        total = 0
        
        # Iterate through the intervals starting from the second one
        for interval in nums[1:]:
            start, end = interval[0], interval[1]
            if start <= current_end:
                # If there is overlap or adjacency, merge the intervals
                current_end = max(current_end, end)
            else:
                # No overlap, add the length of the current merged interval to total
                total += current_end - current_start + 1
                # Start a new merged interval
                current_start = start
                current_end = end
        
        # Add the length of the last merged interval
        total += current_end - current_start + 1
        
        return total