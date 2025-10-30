class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(start_pos, direction, nums_copy):
            curr = start_pos
            n = len(nums_copy)
            
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    # Move in current direction
                    curr += direction
                else:
                    # Decrement the value
                    nums_copy[curr] -= 1
                    # Reverse direction
                    direction = -direction
                    # Move in new direction
                    curr += direction
            
            # Check if all elements are 0
            return all(x == 0 for x in nums_copy)
        
        count = 0
        n = len(nums)
        
        # Find all starting positions where nums[i] == 0
        for i in range(n):
            if nums[i] == 0:
                # Try starting with direction left (-1)
                if simulate(i, -1, nums[:]):
                    count += 1
                
                # Try starting with direction right (1)
                if simulate(i, 1, nums[:]):
                    count += 1
        
        return count