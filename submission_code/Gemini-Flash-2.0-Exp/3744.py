class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        total_operations = 0
        for query in queries:
            l, r = query
            nums = list(range(l, r + 1))
            n = len(nums)
            operations = 0
            while any(num != 0 for num in nums):
                operations += 1
                
                indices_to_reduce = []
                non_zero_indices = [i for i, num in enumerate(nums) if num != 0]
                
                if len(non_zero_indices) >= 2:
                    indices_to_reduce = non_zero_indices[:2]
                elif len(non_zero_indices) == 1:
                    indices_to_reduce = non_zero_indices * 2
                else:
                    break
                
                
                if len(indices_to_reduce) == 2:
                    a_index = indices_to_reduce[0]
                    b_index = indices_to_reduce[1]
                    
                    a = nums[a_index]
                    b = nums[b_index]
                    
                    nums[a_index] = a // 4
                    nums[b_index] = b // 4
            
            total_operations += operations
        
        return total_operations