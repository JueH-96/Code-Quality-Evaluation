import math
from typing import List

class Solution:
    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0

        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if not subarray:
                    continue

                is_product_equivalent = True
                if len(subarray) > 1:
                    for k in range(len(subarray)):
                        for l in range(k + 1, len(subarray)):
                            if self.gcd(subarray[k], subarray[l]) > 1:
                                is_product_equivalent = False
                                break
                        if not is_product_equivalent:
                            break

                if is_product_equivalent:
                    max_len = max(max_len, len(subarray))

        return max_len