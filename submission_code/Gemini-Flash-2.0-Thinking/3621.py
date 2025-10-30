class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if any(num < k for num in nums):
            return -1

        operations = 0
        while any(num > k for num in nums):
            greater_than_k = sorted(list(set(num for num in nums if num > k)), reverse=True)

            if not greater_than_k:
                break

            if len(greater_than_k) == 1:
                h = k
            else:
                h = greater_than_k[1]

            for i in range(len(nums)):
                if nums[i] > h:
                    nums[i] = h
            operations += 1

        return operations