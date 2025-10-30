class Solution:
    def minOperations(self, k: int) -> int:
        operations = 0
        nums = [1]
        current_sum = 1
        
        while current_sum < k:
            if current_sum * 2 <= k:
                # Duplicate the largest element (greedy approach)
                nums.append(nums[-1])
                current_sum += nums[-1]
                operations += 1
            else:
                # Increase the largest element
                increase = k - current_sum
                nums[-1] += increase
                current_sum += increase
                operations += increase
        
        return operations