class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_val = max(nums) + 2
        f = [0] * (max_val + 2)
        nums.sort()
        ans = 0
        for a in nums:
            cand1 = f[a - 1] + 1
            cand2 = f[a] + 1
            if cand1 > f[a]:
                f[a] = cand1
            if cand2 > f[a + 1]:
                f[a + 1] = cand2
            ans = max(ans, cand1, cand2)
        return ans