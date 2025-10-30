class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        
        # Iterate over all possible combinations of p, q, r, s
        for p in range(n - 6):
            for q in range(p + 2, n - 4):
                for r in range(q + 2, n - 2):
                    for s in range(r + 2, n):
                        # Check if the condition nums[p] * nums[r] == nums[q] * nums[s] holds
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
                            
        return count