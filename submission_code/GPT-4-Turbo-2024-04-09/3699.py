class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate over each possible (p, q, r, s) tuple with the given constraints
        for p in range(n - 6):
            for q in range(p + 3, n - 3):
                for r in range(q + 3, n - 1):
                    for s in range(r + 3, n):
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
                            
        return count