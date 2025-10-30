class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all possible values of p, q, r, s
        for p in range(n):
            for q in range(p + 2, n):  # q - p > 1, so q >= p + 2
                for r in range(q + 2, n):  # r - q > 1, so r >= q + 2
                    for s in range(r + 2, n):  # s - r > 1, so s >= r + 2
                        # Check if nums[p] * nums[r] == nums[q] * nums[s]
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        
        return count