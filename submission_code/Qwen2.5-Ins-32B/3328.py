class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        min_ops = float('inf')
        
        for i in range(1, k):
            # Calculate the number of increments needed to reach i
            increments = i - 1
            # Calculate the number of duplicates needed to reach or exceed k
            duplicates = (k + i - 1) // i - 1
            # Total operations is the sum of increments and duplicates
            total_ops = increments + duplicates
            # Update the minimum operations if the current total is smaller
            min_ops = min(min_ops, total_ops)
        
        return min_ops