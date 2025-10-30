class Solution:
    def minDifference(self, nums: List[int]) -> int:
        import sys
        from bisect import bisect_left, bisect_right
        
        # Find all the non-missing values and their positions
        values = []
        positions = []
        for i, num in enumerate(nums):
            if num != -1:
                values.append(num)
                positions.append(i)
        
        # If there are no non-missing values, return 0 (all can be the same)
        if not values:
            return 0
        
        # If there is only one non-missing value, also return 0 (all can be the same as that value)
        if len(values) == 1:
            return 0
        
        # Calculate the minimum possible difference
        min_diff = sys.maxsize
        
        # Iterate through pairs of adjacent non-missing values
        for i in range(1, len(values)):
            left_value = values[i - 1]
            right_value = values[i]
            left_index = positions[i - 1]
            right_index = positions[i]
            
            # If there are missing values between them
            if right_index - left_index > 1:
                # We need to minimize the maximum difference between left_value, right_value, and any x, y we choose
                # The optimal x, y would be as close as possible to the average of left_value and right_value
                if left_value > right_value:
                    left_value, right_value = right_value, left_value
                
                # The range of values to consider for x and y
                possible_values = range(left_value, right_value + 1)
                
                # Try to find the best x and y in the range
                for x in possible_values:
                    for y in possible_values:
                        if x > y:
                            continue
                        max_diff = max(abs(left_value - x), abs(x - y), abs(y - right_value))
                        min_diff = min(min_diff, max_diff)
            else:
                # Directly compute the difference between adjacent non-missing values
                min_diff = min(min_diff, abs(right_value - left_value))
        
        return min_diff