class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        freq = Counter(nums)
        operations = 0
        
        for count in freq.values():
            if count == 1:
                return -1
                
            # For each count, we want to use as many groups of 3 as possible
            # since this minimizes the total number of operations
            operations += count // 3
            
            # If there are remaining elements after groups of 3
            if count % 3:
                # We need one more operation to handle the remainder
                operations += 1
                
        return operations