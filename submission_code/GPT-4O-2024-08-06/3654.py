from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # First, we will consider applying operation 1 (divide by 2) to the largest numbers
        # as this will have the most significant impact on reducing the sum.
        if op1 > 0:
            # Sort indices by the value of nums[i] in descending order
            indices = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)
            for i in indices:
                if op1 > 0:
                    nums[i] = (nums[i] + 1) // 2
                    op1 -= 1
                else:
                    break

        # Next, we will consider applying operation 2 (subtract k) to the largest numbers
        # that are greater than or equal to k.
        if op2 > 0:
            # Sort indices by the value of nums[i] in descending order
            indices = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)
            for i in indices:
                if op2 > 0 and nums[i] >= k:
                    nums[i] -= k
                    op2 -= 1
                else:
                    break

        # Return the sum of the modified array
        return sum(nums)