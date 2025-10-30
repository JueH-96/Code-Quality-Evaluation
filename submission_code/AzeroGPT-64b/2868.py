class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        s = e = 0
        mn = mx = nums[0]
        res = 0
        while e < n:
            mn = min(mn, nums[e])
            mx = max(mx, nums[e])
            while mx - mn > 2:
                mn, mx = min(mn, nums[s+1]), max(mx, nums[s+1])
                s += 1
            res += e - s + 1
            e += 1
        return res