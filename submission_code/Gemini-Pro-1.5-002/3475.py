class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        
        q = [(tuple(nums), 0)]
        visited = {tuple(nums)}
        
        while q:
            curr_nums, curr_ops = q.pop(0)
            
            if all(x == 1 for x in curr_nums):
                return curr_ops
            
            for i in range(n - 2):
                next_nums = list(curr_nums)
                for j in range(i, i + 3):
                    next_nums[j] = 1 - next_nums[j]
                
                next_nums_tuple = tuple(next_nums)
                if next_nums_tuple not in visited:
                    visited.add(next_nums_tuple)
                    q.append((next_nums_tuple, curr_ops + 1))
        
        return -1