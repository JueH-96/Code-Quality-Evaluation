class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        operations = 0
        
        for freq in count.values():
            if freq == 1:
                return -1
            
            operations += freq // 3
            if freq % 3:
                operations += 1
        
        return operations