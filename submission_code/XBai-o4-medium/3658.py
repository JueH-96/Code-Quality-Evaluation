import itertools
from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def split_into_segments(nums):
            segments = []
            n = len(nums)
            i = 0
            while i < n:
                if nums[i] != -1:
                    i += 1
                else:
                    start = i
                    while i < n and nums[i] == -1:
                        i += 1
                    end = i - 1
                    left_val = nums[start-1] if start > 0 else None
                    right_val = nums[end+1] if end+1 < n else None
                    segments.append( (left_val, right_val) )
            return segments

        def is_possible(D):
            segments = split_into_segments(nums)
            m = len(segments)
            if m == 0:
                max_diff = 0
                for i in range(1, len(nums)):
                    max_diff = max(max_diff, abs(nums[i] - nums[i-1]))
                return max_diff <= D
            if m >= 4:
                for (left_val, right_val) in segments:
                    l1 = left_val - D if left_val is not None else -float('inf')
                    r1 = left_val + D if left_val is not None else float('inf')
                    l2 = right_val - D if right_val is not None else -float('inf')
                    r2 = right_val + D if right_val is not None else float('inf')
                    new_low = max(l1, l2)
                    new_high = min(r1, r2)
                    if new_low > new_high:
                        return False
                return True
            else:
                from itertools import product
                for options in product([0,1,2,3], repeat=m):
                    x_low = -float('inf')
                    x_high = float('inf')
                    y_low = -float('inf')
                    y_high = float('inf')
                    has_xy = False
                    valid = True
                    for i in range(m):
                        left_val, right_val = segments[i]
                        opt = options[i]
                        if opt == 0:
                            if left_val is not None:
                                a = left_val - D
                                b = left_val + D
                                new_x_low = max(x_low, a)
                                new_x_high = min(x_high, b)
                                if new_x_low > new_x_high:
                                    valid = False
                                    break
                                x_low = new_x_low
                                x_high = new_x_high
                            if right_val is not None:
                                a = right_val - D
                                b = right_val + D
                                new_x_low = max(x_low, a)
                                new_x_high = min(x_high, b)
                                if new_x_low > new_x_high:
                                    valid = False
                                    break
                                x_low = new_x_low
                                x_high = new_x_high
                        elif opt == 1:
                            has_xy = True
                            if left_val is not None:
                                a = left_val - D
                                b = left_val + D
                                new_x_low = max(x_low, a)
                                new_x_high = min(x_high, b)
                                if new_x_low > new_x_high:
                                    valid = False
                                    break
                                x_low = new_x_low
                                x_high = new_x_high
                            if right_val is not None:
                                a = right_val - D
                                b = right_val + D
                                new_y_low = max(y_low, a)
                                new_y_high = min(y_high, b)
                                if new_y_low > new_y_high:
                                    valid = False
                                    break
                                y_low = new_y_low
                                y_high = new_y_high
                        elif opt == 2:
                            has_xy = True
                            if left_val is not None:
                                a = left_val - D
                                b = left_val + D
                                new_y_low = max(y_low, a)
                                new_y_high = min(y_high, b)
                                if new_y_low > new_y_high:
                                    valid = False
                                    break
                                y_low = new_y_low
                                y_high = new_y_high
                            if right_val is not None:
                                a = right_val - D
                                b = right_val + D
                                new_x_low = max(x_low, a)
                                new_x_high = min(x_high, b)
                                if new_x_low > new_x_high:
                                    valid = False
                                    break
                                x_low = new_x_low
                                x_high = new_x_high
                        elif opt == 3:
                            if left_val is not None:
                                a = left_val - D
                                b = left_val + D
                                new_y_low = max(y_low, a)
                                new_y_high = min(y_high, b)
                                if new_y_low > new_y_high:
                                    valid = False
                                    break
                                y_low = new_y_low
                                y_high = new_y_high
                            if right_val is not None:
                                a = right_val - D
                                b = right_val + D
                                new_y_low = max(y_low, a)
                                new_y_high = min(y_high, b)
                                if new_y_low > new_y_high:
                                    valid = False
                                    break
                                y_low = new_y_low
                                y_high = new_y_high
                    if not valid:
                        continue
                    if has_xy:
                        if x_low > x_high or y_low > y_high:
                            continue
                        if (x_high - y_low <= D) and (y_high - x_low <= D):
                            return True
                    else:
                        if x_low <= x_high and y_low <= y_high:
                            return True
                return False

        low = 0
        high = 10**9
        if all(num == -1 for num in nums):
            return 0
        while low < high:
            mid = (low + high) // 2
            if is_possible(mid):
                high = mid
            else:
                low = mid + 1
        return low