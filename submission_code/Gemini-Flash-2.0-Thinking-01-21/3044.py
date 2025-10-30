class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected_numbers = set()
        operations = 0
        count = 0
        n = len(nums)
        for i in range(n - 1, -1, -1):
            removed_num = nums[i]
            operations += 1
            if 1 <= removed_num <= k and removed_num not in collected_numbers:
                collected_numbers.add(removed_num)
                count += 1
            if count == k:
                return operations
        return operations