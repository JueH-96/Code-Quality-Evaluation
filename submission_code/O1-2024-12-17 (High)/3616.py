class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)

        # Helper function to simulate the process,
        # returns True if all elements end up as 0.
        def can_zero_out(arr, start, direction):
            arr_copy = arr[:]
            curr = start
            d = direction
            while 0 <= curr < n:
                if arr_copy[curr] == 0:
                    curr += d
                else:
                    arr_copy[curr] -= 1
                    d = -d  # reverse direction
                    curr += d
            return all(x == 0 for x in arr_copy)

        # Try each position where nums[i] == 0, with both left and right directions.
        count = 0
        for i in range(n):
            if nums[i] == 0:
                if can_zero_out(nums, i, -1):
                    count += 1
                if can_zero_out(nums, i, 1):
                    count += 1
        return count