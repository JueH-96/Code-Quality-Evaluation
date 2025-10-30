from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0

        for k in range(n // 2, 0, -1):
            # The score of the operations will be determined by the first operation
            # if we perform k operations.
            target_score = nums[0] + nums[1]
            possible = True
            temp_nums = list(nums)
            num_ops = 0

            # Simulate k operations
            for _ in range(k):
                if len(temp_nums) < 2:
                    possible = False
                    break

                current_score = temp_nums[0] + temp_nums[1]
                if current_score == target_score:
                    num_ops += 1
                    temp_nums = temp_nums[2:]
                else:
                    possible = False
                    break

            if possible and num_ops == k:
                return k

        return 0