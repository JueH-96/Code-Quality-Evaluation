class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        zero_positions = [i for i, val in enumerate(nums) if val == 0]
        directions = [-1, 1]
        
        def simulate(start_index, direction):
            arr = nums[:]  # make a copy to avoid mutating the original
            curr = start_index
            dir_ = direction
            while 0 <= curr < n:
                if arr[curr] == 0:
                    # Just move in the same direction
                    curr += dir_
                else:
                    # Decrement and reverse direction
                    arr[curr] -= 1
                    dir_ = -dir_
                    curr += dir_
            return all(x == 0 for x in arr)
        
        valid_count = 0
        for pos in zero_positions:
            for d in directions:
                if simulate(pos, d):
                    valid_count += 1
        
        return valid_count