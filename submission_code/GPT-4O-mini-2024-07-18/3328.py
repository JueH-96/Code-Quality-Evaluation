class Solution:
    def minOperations(self, k: int) -> int:
        operations = 0
        current_sum = 1
        current_value = 1
        
        while current_sum < k:
            if current_sum + current_value >= k:
                operations += k - current_sum
                break
            
            current_sum += current_value
            operations += 1
            current_value += 1
            
        return operations