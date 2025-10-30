from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def is_possible(d):
            x_min, x_max = 1, int(1e9)
            y_min, y_max = 1, int(1e9)
            prev = None
            for i in range(len(nums)):
                if nums[i] == -1:
                    if prev is not None:
                        if prev == 'x':
                            x_min = max(x_min, nums[i-1] - d)
                            x_max = min(x_max, nums[i-1] + d)
                        elif prev == 'y':
                            y_min = max(y_min, nums[i-1] - d)
                            y_max = min(y_max, nums[i-1] + d)
                    prev = 'x'  # Assume we place x for now
                else:
                    if prev == 'x':
                        x_min = max(x_min, nums[i] - d)
                        x_max = min(x_max, nums[i] + d)
                    elif prev == 'y':
                        y_min = max(y_min, nums[i] - d)
                        y_max = min(y_max, nums[i] + d)
                    prev = 'val'
            # Check if there exist x and y in their ranges with |x - y| <= d
            x_range = range(x_min, x_max + 1)
            y_range = range(y_min, y_max + 1)
            # Check if x and y can be chosen such that |x - y| <= d
            if x_range and y_range:
                x_start = x_range[0]
                y_start = y_range[0]
                if abs(x_start - y_start) <= d:
                    return True
                if x_start + d >= y_start and y_start + d >= x_start:
                    return True
            return False
        
        # Binary search on d
        low = 0
        high = int(1e9)
        while low < high:
            mid = (low + high) // 2
            if is_possible(mid):
                high = mid
            else:
                low = mid + 1
        return low