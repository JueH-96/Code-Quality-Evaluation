class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Count how many are missing
        missing_count = nums.count(-1)
        
        # If all are missing, we can fill them all with the same value => difference = 0
        if missing_count == n:
            return 0
        
        # We will track:
        #   1) M = maximum absolute difference between adjacent known-known elements
        #   2) B = set of known values that are adjacent to at least one missing
        
        M = 0
        B = set()
        
        for i in range(n - 1):
            a, b = nums[i], nums[i+1]
            
            # If both are known, update M
            if a != -1 and b != -1:
                M = max(M, abs(a - b))
            
            # If one is known and the other is missing, add the known one to B
            if a != -1 and b == -1:
                B.add(a)
            if a == -1 and b != -1:
                B.add(b)
        
        # If there are no missing elements at all, the answer is just M
        if missing_count == 0:
            return M
        
        # If B is empty but not all are missing, it means we have only known elements
        # (handled above) or the entire array is missing (handled above). So no new case needed.
        if not B:
            # This would happen if everything is missing (covered above) or no adjacency to missing
            # but that implies no missing => already handled. So just return M here.
            return M
        
        # Otherwise, find min and max among known values adjacent to missing
        minB = min(B)
        maxB = max(B)
        
        # The best we can do is to place x and y around the midpoint of [minB, maxB].
        # The effective boundary difference is roughly ceil((maxB - minB) / 2).
        # We'll do (maxB - minB + 1)//2 for an integer ceiling.
        boundary_diff = (maxB - minB + 1) // 2
        
        # The final answer is the maximum of M and that boundary difference
        return max(M, boundary_diff)