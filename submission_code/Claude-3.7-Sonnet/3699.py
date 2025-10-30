class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all valid combinations
        for p in range(n - 6):  # Lowest p can be 0, highest can be n-7
            for q in range(p + 2, n - 4):  # Ensure q-p>1 and leave room for r and s
                for r in range(q + 2, n - 2):  # Ensure r-q>1 and leave room for s
                    for s in range(r + 2, n):  # Ensure s-r>1
                        # Check if the equation is satisfied
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        
        return count