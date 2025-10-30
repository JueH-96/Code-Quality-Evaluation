class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each element
        freq = Counter(nums)
        
        total_operations = 0
        
        for count in freq.values():
            if count == 1:
                return -1
            total_operations += (count + 2) // 3
        
        return total_operations