class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each number
        count = Counter(nums)
        
        total_operations = 0
        
        for freq in count.values():
            # If frequency is 1, impossible to remove
            if freq == 1:
                return -1
            
            # Calculate minimum operations for this frequency
            if freq % 3 == 0:
                # Use only operations of size 3
                total_operations += freq // 3
            elif freq % 3 == 1:
                # Use (freq-4)/3 operations of size 3 and 2 operations of size 2
                # This works because 4 = 2 + 2
                total_operations += (freq - 4) // 3 + 2
            else:  # freq % 3 == 2
                # Use (freq-2)/3 operations of size 3 and 1 operation of size 2
                total_operations += (freq - 2) // 3 + 1
        
        return total_operations