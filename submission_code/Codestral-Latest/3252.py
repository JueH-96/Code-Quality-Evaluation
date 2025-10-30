class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_increasing(arr):
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))

        n = len(nums)
        count = 0

        for i in range(n):
            for j in range(i, n):
                # Create a new array excluding the subarray nums[i:j+1]
                new_arr = nums[:i] + nums[j+1:]
                if is_increasing(new_arr):
                    count += 1

        return count