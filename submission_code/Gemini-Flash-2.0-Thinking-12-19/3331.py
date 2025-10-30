class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        while True:
            all_greater_equal_k = True
            for num in nums:
                if num < k:
                    all_greater_equal_k = False
                    break
            if all_greater_equal_k:
                break

            if not nums:
                break

            min_val = min(nums)
            min_index = -1
            for i in range(len(nums)):
                if nums[i] == min_val:
                    min_index = i
                    break
            nums.pop(min_index)
            operations += 1
        return operations