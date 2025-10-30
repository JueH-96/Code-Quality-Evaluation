class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        operations = 0
        for count in counts.values():
            if count == 1:
                return -1
            
            operations += (count + 2) // 3
        
        return operations