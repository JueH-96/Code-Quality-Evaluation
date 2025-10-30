class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0

        operations = 0
        current_sum = 1
        nums = [1]

        while current_sum < k:
            # Always duplicate the last element
            nums.append(nums[-1])
            operations += 1
            current_sum = sum(nums)

            if current_sum >= k:
                break

            # Increase the last element until it's optimal to duplicate again
            while nums[-1] * 2 <= nums[0] * (len(nums) + 1):
                nums[-1] += 1
                operations += 1
                current_sum = sum(nums)

                if current_sum >= k:
                    break

        return operations