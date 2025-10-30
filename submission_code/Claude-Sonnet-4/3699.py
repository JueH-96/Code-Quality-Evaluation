class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all possible p values
        for p in range(n - 6):  # Need at least 6 more elements after p
            # q must be at least p + 2 (at least one element between p and q)
            for q in range(p + 2, n - 4):  # Need at least 4 more elements after q
                # r must be at least q + 2 (at least one element between q and r)
                for r in range(q + 2, n - 2):  # Need at least 2 more elements after r
                    # s must be at least r + 2 (at least one element between r and s)
                    for s in range(r + 2, n):
                        # Check if the product condition is satisfied
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        
        return count