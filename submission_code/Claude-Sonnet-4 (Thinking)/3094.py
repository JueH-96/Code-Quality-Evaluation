class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        total_ops = 0
        
        for freq in count.values():
            if freq == 1:
                return -1
            total_ops += (freq + 2) // 3
        
        return total_ops