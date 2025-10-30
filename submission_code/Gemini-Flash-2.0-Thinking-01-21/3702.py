from typing import List
import math

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        prime_factors = [2, 3, 5, 7]
        factorizations = [{}, {2: 1}, {3: 1}, {2: 2}, {5: 1}, {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}, {2: 1, 5: 1}]
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                is_product_equivalent = True
                for p in prime_factors:
                    exponents = []
                    for num in subarray:
                        factorization = factorizations[num]
                        exponent = factorization.get(p, 0)
                        exponents.append(exponent)
                    if len(set(exponents)) > 2:
                        is_product_equivalent = False
                        break
                if is_product_equivalent:
                    max_len = max(max_len, len(subarray))
        return max_len

def gcd_array(arr):
    if not arr:
        return 0
    result = arr[0]
    for i in range(1, len(arr)):
        result = math.gcd(result, arr[i])
    return result

def lcm_array(arr):
    if not arr:
        return 1
    result = arr[0]
    for i in range(1, len(arr)):
        result = (result * arr[i]) // math.gcd(result, arr[i])
    return result

def product_array(arr):
    prod = 1
    for x in arr:
        prod *= x
    return prod

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                prod_val = product_array(subarray)
                gcd_val = gcd_array(subarray)
                lcm_val = lcm_array(subarray)
                if prod_val == lcm_val * gcd_val:
                    max_len = max(max_len, len(subarray))
        return max_len