class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # First, check if any two same elements are at least x apart
        from collections import defaultdict
        index_map = defaultdict(list)
        for i, num in enumerate(nums):
            index_map[num].append(i)
        min_diff = float('inf')
        # Check for same values with indices at least x apart
        for key in index_map:
            indices = index_map[key]
            if len(indices) >= 2:
                # Sort the indices
                indices.sort()
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if indices[j] - indices[i] >= x:
                            return 0
        # If no such pair found, proceed
        # Sort the array along with original indices
        sorted_nums = sorted((nums[i], i) for i in range(len(nums)))
        n = len(sorted_nums)
        min_diff = float('inf')
        # For each element, find the closest in the entire sorted array
        for i in range(n):
            current_val, current_idx = sorted_nums[i]
            # Binary search for the position where current_val would fit
            left = 0
            right = n - 1
            while left <= right:
                mid = (left + right) // 2
                if sorted_nums[mid][0] < current_val:
                    left = mid + 1
                else:
                    right = mid - 1
            # Check the elements at 'left' and 'right' positions
            for j in [right, left]:
                if 0 <= j < n:
                    j_val, j_idx = sorted_nums[j]
                    if abs(j_idx - current_idx) >= x:
                        diff = abs(current_val - j_val)
                        if diff < min_diff:
                            min_diff = diff
        return min_diff if min_diff != float('inf') else 0