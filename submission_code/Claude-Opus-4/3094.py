class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each number
        freq = Counter(nums)
        
        total_operations = 0
        
        # Check each unique number
        for count in freq.values():
            # If count is 1, it's impossible to remove
            if count == 1:
                return -1
            
            # Calculate minimum operations needed
            # We want to use as many 3s as possible to minimize operations
            # The formula is ceil(count/3)
            total_operations += (count + 2) // 3
        
        return total_operations