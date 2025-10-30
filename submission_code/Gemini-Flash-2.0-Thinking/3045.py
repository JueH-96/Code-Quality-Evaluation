class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)

        def is_sorted(arr):
            for i in range(len(arr) - 1):
                if arr[i] > arr[i + 1]:
                    return False
            return True

        if is_sorted(nums):
            return 0

        shifted_nums = list(nums)
        for shifts in range(1, n):
            # Perform a right shift
            last = shifted_nums[-1]
            shifted_nums.insert(0, last)
            shifted_nums.pop()

            if is_sorted(shifted_nums):
                return shifts

        return -1