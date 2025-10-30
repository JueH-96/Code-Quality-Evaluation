from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        max_count = 0
        max_num = nums[0]
        
        # Find the most frequent number in the array
        for num in nums:
            if nums.count(num) > max_count:
                max_count = nums.count(num)
                max_num = num
        
        # If all elements are the same, return 0
        if max_count == n:
            return 0
        
        # Initialize the result (minimum seconds)
        res = 0
        
        # Perform BFS
        queue = [nums]
        visited = {tuple(nums)}
        
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                curr = queue.pop(0)
                # Check if all elements are the same
                if len(set(curr)) == 1:
                    return res
                
                # Generate all possible next states
                for i in range(n):
                    next_state1 = curr[:i] + [curr[(i - 1) % n]] + curr[i + 1:]
                    next_state2 = curr[:i] + [curr[(i + 1) % n]] + curr[i + 1:]
                    
                    # Add next states to the queue if not visited
                    if tuple(next_state1) not in visited:
                        queue.append(next_state1)
                        visited.add(tuple(next_state1))
                    if tuple(next_state2) not in visited:
                        queue.append(next_state2)
                        visited.add(tuple(next_state2))
            
            # Increment the result (minimum seconds)
            res += 1