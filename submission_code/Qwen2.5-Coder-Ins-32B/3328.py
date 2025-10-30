class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        operations = 0
        current_value = 1
        
        while current_value * (current_value + 1) // 2 < k:
            current_value += 1
            operations += 1
        
        target_sum = current_value * (current_value + 1) // 2
        if target_sum >= k:
            return operations + (current_value - (target_sum - k))
        
        return operations