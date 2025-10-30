class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i+1] for i in range(len(arr)-1))
        for a in range(n - 2*k + 1):
            first_subarray = nums[a:a+k]
            second_subarray = nums[a+k:a+2*k]
            if is_strictly_increasing(first_subarray) and is_strictly_increasing(second_subarray):
                return True
        return False