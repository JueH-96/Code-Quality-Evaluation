class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        total_operations = 0
        
        for l, r in queries:
            nums = list(range(l, r+1))
            operations = 0
            
            while any(num > 0 for num in nums):
                a, b = None, None
                min_val = float('inf')
                
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        if nums[i] > 0 and nums[j] > 0:
                            new_a = nums[i] // 4
                            new_b = nums[j] // 4
                            if new_a + new_b < min_val:
                                a, b = i, j
                                min_val = new_a + new_b
                
                if a is not None and b is not None:
                    nums[a] = nums[a] // 4
                    nums[b] = nums[b] // 4
                    operations += 1
                else:
                    break
            
            total_operations += operations
        
        return total_operations