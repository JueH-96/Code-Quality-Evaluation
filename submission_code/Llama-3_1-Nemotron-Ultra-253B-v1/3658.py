from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Compute the fixed maximum difference between non--1 adjacent elements
        fixed_max = 0
        n = len(nums)
        for i in range(n-1):
            if nums[i] != -1 and nums[i+1] != -1:
                fixed_max = max(fixed_max, abs(nums[i] - nums[i+1]))
        
        # Generate groups of consecutive -1s
        groups = []
        i = 0
        while i < n:
            if nums[i] == -1:
                start = i
                while i < n and nums[i] == -1:
                    i += 1
                end = i - 1
                # Determine L and R
                L = nums[start-1] if start > 0 else None
                R = nums[end+1] if end < n-1 else None
                groups.append( (start, end, L, R) )
            else:
                i += 1
        
        # Binary search for the minimal G
        left = fixed_max
        right = 10**18  # A large enough upper bound
        
        # Handle the case where there are no groups
        if not groups:
            return fixed_max
        
        # Function to check if G is feasible
        def is_feasible(G):
            ranges = []
            has_multiple = False
            for group in groups:
                s, e, L, R = group
                length = e - s + 1
                if length >= 2:
                    has_multiple = True
                    # Add L and R ranges
                    if L is not None:
                        ranges.append( (L - G, L + G) )
                    if R is not None:
                        ranges.append( (R - G, R + G) )
                else:
                    # Single -1, compute required range
                    req_range = []
                    if L is not None and R is not None:
                        lower = max(L - G, R - G)
                        upper = min(L + G, R + G)
                        if lower > upper:
                            return False
                        req_range = (lower, upper)
                    elif L is not None:
                        req_range = (L - G, L + G)
                    elif R is not None:
                        req_range = (R - G, R + G)
                    else:
                        # All elements are -1, no constraint
                        continue
                    ranges.append(req_range)
            
            # Check if all ranges can be covered by x and y, with |x-y| <= G if has_multiple
            if not ranges:
                return True
            # Sort ranges by their end
            ranges.sort(key=lambda x: x[1])
            # Try to find x and y
            x = ranges[0][1]
            # Find the first range not covered by x
            for i in range(len(ranges)):
                if ranges[i][0] > x:
                    break
            else:
                # All ranges covered by x
                return True
            # Need to choose y to cover ranges[i] and beyond
            y = ranges[i][1]
            if has_multiple and y - x > G:
                return False
            # Check if y covers all remaining ranges
            for j in range(i, len(ranges)):
                if ranges[j][0] > y:
                    return False
            return True
        
        # Binary search
        answer = fixed_max
        while left <= right:
            mid = (left + right) // 2
            if is_feasible(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer