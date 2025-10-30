class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        def dfs(curr, direction):
            nonlocal count
            
            if curr < 0 or curr >= n:
                return
            
            if nums[curr] == 0:
                new_curr = curr + 1 if direction == 'right' else curr - 1
                dfs(new_curr, direction)
            else:
                nums[curr] -= 1
                new_direction = 'right' if direction == 'left' else 'left'
                new_curr = curr + 1 if new_direction == 'right' else curr - 1
                dfs(new_curr, new_direction)
                
                if all(num == 0 for num in nums):
                    count += 1
                
                nums[curr] += 1
        
        for i in range(n):
            if nums[i] == 0:
                dfs(i, 'right')
                dfs(i, 'left')
        
        return count