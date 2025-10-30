from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Identify the positions of -1
        missing_indices = [i for i, num in enumerate(nums) if num == -1]
        if not missing_indices:
            # No missing elements, calculate the maximum difference
            max_diff = 0
            for i in range(1, len(nums)):
                max_diff = max(max_diff, abs(nums[i] - nums[i-1]))
            return max_diff
        
        # Determine the possible values for x and y
        # We need to find the range of possible values for x and y
        # The values should be such that when replacing -1, the differences are minimized
        
        # For each missing index, determine the possible range based on its neighbors
        possible_values = []
        for idx in missing_indices:
            left = nums[idx - 1] if idx > 0 else None
            right = nums[idx + 1] if idx < len(nums) - 1 else None
            if left is not None and right is not None:
                # The value should be between left and right to minimize the difference
                # So, the possible value is in [min(left, right), max(left, right)]
                possible_values.append((min(left, right), max(left, right)))
            elif left is not None:
                # Only left neighbor, the value can be any, but to minimize the difference, it should be close to left
                possible_values.append((left, left))
            elif right is not None:
                # Only right neighbor, the value can be any, but to minimize the difference, it should be close to right
                possible_values.append((right, right))
            else:
                # No neighbors, can be any value, but to minimize the difference, it should be the same as other missing values
                possible_values.append((None, None))
        
        # Now, we need to find x and y such that for each missing index, the chosen value is either x or y
        # and the maximum difference is minimized
        
        # To minimize the maximum difference, x and y should be as close as possible to the required values
        # So, we can consider the possible ranges and find the best x and y
        
        # Since the number of missing indices can be large, we need an efficient way to find x and y
        # We can use binary search to find the optimal x and y
        
        # The idea is to find the minimum possible maximum difference by trying different x and y values
        # We can perform a binary search on the possible maximum difference
        
        # The possible maximum difference ranges from 0 to the maximum possible difference in the array
        # The maximum possible difference is the difference between the maximum and minimum values in the array
        
        # First, find the minimum and maximum values in the array, excluding -1
        non_missing = [num for num in nums if num != -1]
        if not non_missing:
            # All elements are missing, choose x and y to be the same
            return 0
        min_val = min(non_missing)
        max_val = max(non_missing)
        
        # The possible maximum difference ranges from 0 to (max_val - min_val)
        low = 0
        high = max_val - min_val
        
        # Function to check if a given maximum difference is possible
        def is_possible(max_diff):
            # We need to find x and y such that for each missing index, the chosen value is either x or y
            # and the differences with neighbors are <= max_diff
            # We can try to assign x and y to the missing indices in a way that satisfies the constraints
            
            # For each missing index, determine the possible range for the value based on its neighbors and max_diff
            # Then, find x and y that can cover all these ranges
            
            # Initialize x and y as None
            x = None
            y = None
            
            for idx in missing_indices:
                left = nums[idx - 1] if idx > 0 else None
                right = nums[idx + 1] if idx < len(nums) - 1 else None
                if left is not None and right is not None:
                    # The value should be between left and right, and the differences with left and right should be <= max_diff
                    # So, the value should be in [left - max_diff, left + max_diff] and [right - max_diff, right + max_diff]
                    # The intersection of these two ranges is the possible range for the value
                    lower = max(left - max_diff, right - max_diff)
                    upper = min(left + max_diff, right + max_diff)
                    if lower > upper:
                        return False
                    # Now, we need to assign x or y to this range
                    if x is None and y is None:
                        # Assign x to the lower bound and y to the upper bound
                        x = lower
                        y = upper
                    else:
                        # Check if x or y can be assigned to this range
                        # x must be in [lower, upper] or y must be in [lower, upper]
                        if not (lower <= x <= upper or lower <= y <= upper):
                            return False
                elif left is not None:
                    # Only left neighbor, the value should be in [left - max_diff, left + max_diff]
                    lower = left - max_diff
                    upper = left + max_diff
                    if x is None and y is None:
                        x = lower
                        y = upper
                    else:
                        if not (lower <= x <= upper or lower <= y <= upper):
                            return False
                elif right is not None:
                    # Only right neighbor, the value should be in [right - max_diff, right + max_diff]
                    lower = right - max_diff
                    upper = right + max_diff
                    if x is None and y is None:
                        x = lower
                        y = upper
                    else:
                        if not (lower <= x <= upper or lower <= y <= upper):
                            return False
                else:
                    # No neighbors, can be any value, but to minimize the difference, it should be the same as other missing values
                    # So, x and y should be the same
                    if x is None and y is None:
                        x = 0
                        y = 0
                    else:
                        if x != y:
                            return False
            return True
        
        # Perform binary search to find the minimum possible maximum difference
        result = high
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        return result