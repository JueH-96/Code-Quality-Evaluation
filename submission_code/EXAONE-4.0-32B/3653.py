class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        P = [0] * (n + 1)
        for i in range(1, n + 1):
            P[i] = P[i - 1] + nums[i - 1]
        
        min_pref = [10**18] * k
        min_pref[0] = 0
        ans = -10**18
        
        for i in range(1, n + 1):
            r = i % k
            if i >= k:
                if P[i - k] < min_pref[r]:
                    min_pref[r] = P[i - k]
                candidate = P[i] - min_pref[r]
                if candidate > ans:
                    ans = candidate
        return ans