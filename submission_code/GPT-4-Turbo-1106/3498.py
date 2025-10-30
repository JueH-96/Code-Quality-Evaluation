class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        n = len(nums)
        pairs = Counter()
        
        for i in range(n // 2):
            pairs[nums[i], nums[n - i - 1]] += 1
        
        max_freq = 0
        for (x, y), freq in pairs.items():
            if x == y:
                max_freq += freq
            else:
                max_freq = max(max_freq, freq + pairs[y, x])
        
        return n // 2 - max_freq