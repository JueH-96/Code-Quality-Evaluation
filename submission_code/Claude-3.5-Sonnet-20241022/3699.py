class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Check all possible combinations of p, q, r, s
        for p in range(n-3):
            for q in range(p+2, n-2):  # At least one element between p and q
                for r in range(q+2, n-1):  # At least one element between q and r
                    for s in range(r+2, n):  # At least one element between r and s
                        # Check if the condition nums[p] * nums[r] == nums[q] * nums[s] is satisfied
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        
        return count