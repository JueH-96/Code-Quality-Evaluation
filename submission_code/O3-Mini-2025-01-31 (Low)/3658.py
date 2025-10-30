from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # maxDiff records maximum absolute difference of two adjacent non-missing numbers.
        maxDiff = 0
        # For any valid candidate where one side is missing and one side is not,
        # we record the known number value. We'll then take the minimum and maximum of these.
        cand_min = 10**9 + 1
        cand_max = 0
        
        for i in range(n):
            if nums[i] != -1:
                # Check left neighbor if it exists and is missing.
                if i - 1 >= 0 and nums[i-1] == -1:
                    cand_min = min(cand_min, nums[i])
                    cand_max = max(cand_max, nums[i])
                # Check right neighbor if it exists and is missing.
                if i + 1 < n and nums[i+1] == -1:
                    cand_min = min(cand_min, nums[i])
                    cand_max = max(cand_max, nums[i])
            # Also update the maxDiff for two consecutive known numbers.
            if i > 0 and nums[i] != -1 and nums[i-1] != -1:
                maxDiff = max(maxDiff, abs(nums[i] - nums[i-1]))
        
        # If no boundaries exist between known and missing,
        # we can simply choose any pair (x, y) to fill missing such that adjacent differences are zero.
        if cand_min == 10**9 + 1:
            return maxDiff
        
        # The best we can do for transitions that involve replaced missing values is to pick a value
        # in between. In fact, the minimal worst-case difference we can get from bridging a gap
        # is (cand_max - cand_min + 1) // 2. This is because we can independently assign x or y.
        candidate_diff = (cand_max - cand_min + 1) // 2
        
        return max(maxDiff, candidate_diff)
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minDifference([1,2,-1,10,8]))      # Output: 4
    print(sol.minDifference([-1,-1,-1]))         # Output: 0
    print(sol.minDifference([-1,10,-1,8]))         # Output: 1