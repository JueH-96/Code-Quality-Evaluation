class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all valid combinations of indices
        for p in range(n - 6):  # p can be at most n-7
            for q in range(p + 2, n - 4):  # q must be at least p+2 and at most n-5
                for r in range(q + 2, n - 2):  # r must be at least q+2 and at most n-3
                    # For efficiency, check if we can find valid s values
                    # We need nums[s] = (nums[p] * nums[r]) / nums[q]
                    if (nums[p] * nums[r]) % nums[q] == 0:
                        target = (nums[p] * nums[r]) // nums[q]
                        # Check all valid s values
                        for s in range(r + 2, n):  # s must be at least r+2
                            if nums[s] == target:
                                count += 1
        
        return count