class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        zero_positions = [i for i, val in enumerate(nums) if val == 0]
        
        def all_zero(arr):
            return all(x == 0 for x in arr)
        
        def simulate(start, direction):
            # Make a copy of nums to simulate on
            arr = nums[:]
            curr = start
            d = direction
            while 0 <= curr < n:
                if arr[curr] == 0:
                    # Move in the current direction
                    curr += d
                else:
                    # Decrement
                    arr[curr] -= 1
                    # Reverse direction
                    d = -d
                    # Then move
                    curr += d
            return all_zero(arr)
        
        count = 0
        for pos in zero_positions:
            # Try left or right
            if simulate(pos, -1):
                count += 1
            if simulate(pos, 1):
                count += 1
        
        return count