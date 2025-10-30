class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        def is_sorted(arr):
            for i in range(len(arr) - 1):
                if arr[i] > arr[i+1]:
                    return False
            return True

        if is_sorted(nums):
            return 0

        temp_nums = list(nums) # Create a copy to avoid modifying original nums during shifts
        for shifts in range(1, n + 1):
            last_element = temp_nums[-1]
            for i in range(n - 1, 0, -1):
                temp_nums[i] = temp_nums[i-1]
            temp_nums[0] = last_element
            if is_sorted(temp_nums):
                return shifts
        return -1