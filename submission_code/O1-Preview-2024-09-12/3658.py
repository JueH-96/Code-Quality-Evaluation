class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff_known = 0
        min_adjacent = None
        max_adjacent = None
        
        # First pass to compute max_diff_known and collect adjacents
        for i in range(n - 1):
            a, b = nums[i], nums[i + 1]
            if a != -1 and b != -1:
                diff = abs(a - b)
                if diff > max_diff_known:
                    max_diff_known = diff
            if a == -1 and b != -1:
                if min_adjacent is None or b < min_adjacent:
                    min_adjacent = b
                if max_adjacent is None or b > max_adjacent:
                    max_adjacent = b
            if a != -1 and b == -1:
                if min_adjacent is None or a < min_adjacent:
                    min_adjacent = a
                if max_adjacent is None or a > max_adjacent:
                    max_adjacent = a
        
        if min_adjacent is None:
            # All elements are -1 or no adjacents, any number replaces -1 with zero max difference
            return 0
        else:
            D_candidate = (max_adjacent - min_adjacent + 1) // 2
            D = max(max_diff_known, D_candidate)
            K = (min_adjacent + max_adjacent) // 2
            # Replace all -1 with K and compute max difference
            prev = None
            max_diff = 0
            for i in range(n):
                if nums[i] == -1:
                    nums[i] = K
                if prev is not None:
                    diff = abs(nums[i] - prev)
                    if diff > max_diff:
                        max_diff = diff
                prev = nums[i]
            # The maximum difference should not exceed D
            return max_diff if max_diff > D else D