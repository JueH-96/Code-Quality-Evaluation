class Solution:
    def is_strictly_increasing(self, arr):
        if len(arr) <= 1:
            return True
        for k in range(len(arr) - 1):
            if arr[k] >= arr[k+1]:
                return False
        return True

    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                # Subarray to remove: nums[i:j+1]
                left_part = nums[:i]
                right_part = nums[j+1:]

                is_left_increasing = self.is_strictly_increasing(left_part)
                is_right_increasing = self.is_strictly_increasing(right_part)

                if is_left_increasing and is_right_increasing:
                    if left_part and right_part and left_part[-1] >= right_part[0]:
                        continue
                    count += 1
        return count