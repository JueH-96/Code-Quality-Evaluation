class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        prefix_k = prefix.index(k)
        suffix = prefix[::-1]
        suffix_k = suffix.index(k)
        return min(prefix_k, n-suffix_k)