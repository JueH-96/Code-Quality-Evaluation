class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        # 1) Track the maximum absolute difference between consecutive known values
        max_adj_known = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                max_adj_known = max(max_adj_known, abs(nums[i] - nums[i+1]))
        
        # 2) Collect all known values that are adjacent to missing positions
        boundary = []
        have_missing = False
        for i in range(n):
            if nums[i] == -1:
                have_missing = True
            else:
                if i > 0 and nums[i-1] == -1:
                    boundary.append(nums[i])
                if i < n - 1 and nums[i+1] == -1:
                    boundary.append(nums[i])
        
        # If there are no boundary values and we do have missing,
        # then the entire array can be filled with the same value => 0 difference
        if not boundary:
            if have_missing:
                return 0
            # Otherwise, no missing at all => return existing max_adj_known
            return max_adj_known
        
        # 3) Find the min and max among boundary values
        min_b = min(boundary)
        max_b = max(boundary)
        
        # 4) We will pick (x, y) around the midpoint of the boundary range.
        #    The difference to any boundary value is at most (max_b - min_b + 1)//2 (ceiling of half).
        diff_half = (max_b - min_b + 1) // 2
        
        # 5) The result is the maximum of the existing known adjacency gap
        #    and that half-difference.
        return max(max_adj_known, diff_half)