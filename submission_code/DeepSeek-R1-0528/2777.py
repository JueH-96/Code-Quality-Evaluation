class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [0] * n
        seen_prefix = set()
        for i in range(n):
            seen_prefix.add(nums[i])
            prefix[i] = len(seen_prefix)
        
        suf = [0] * (n + 1)
        seen_suf = set()
        suf[n] = 0
        for i in range(n - 1, -1, -1):
            seen_suf.add(nums[i])
            suf[i] = len(seen_suf)
        
        diff = [0] * n
        for i in range(n):
            diff[i] = prefix[i] - suf[i + 1]
        
        return diff