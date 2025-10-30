class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(start_pos, direction):
            # Make a copy of nums to simulate
            arr = nums[:]
            curr = start_pos
            dir_val = direction  # 1 for right, -1 for left
            
            while 0 <= curr < len(arr):
                if arr[curr] == 0:
                    # Move in current direction
                    curr += dir_val
                else:
                    # arr[curr] > 0
                    # Decrement the value
                    arr[curr] -= 1
                    # Reverse direction
                    dir_val = -dir_val
                    # Take a step in new direction
                    curr += dir_val
            
            # Check if all elements are 0
            return all(x == 0 for x in arr)
        
        count = 0
        n = len(nums)
        
        # Find all starting positions (where nums[i] == 0)
        for i in range(n):
            if nums[i] == 0:
                # Try both directions
                # Direction: -1 for left, 1 for right
                if simulate(i, -1):  # left
                    count += 1
                if simulate(i, 1):   # right
                    count += 1
        
        return count