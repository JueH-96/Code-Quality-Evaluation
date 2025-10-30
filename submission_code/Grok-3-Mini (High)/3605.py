from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        result = []
        for num in nums:
            if num % 2 == 0:  # Even number, only possible for prime 2, which is impossible
                result.append(-1)
            else:  # Odd number, find the smallest q such that the q-th bit is 0
                q = 1  # Start from bit 1 since bit 0 is always 1 for odd numbers
                while (num & (1 << q)) != 0:
                    q += 1
                # q is the position of the least significant 0 bit
                p_max = q - 1
                ans_val = num - (1 << p_max)
                result.append(ans_val)
        return result