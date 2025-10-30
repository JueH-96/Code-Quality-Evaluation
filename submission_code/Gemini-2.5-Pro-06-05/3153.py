from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum sum of squares of k elements after applying a specific
        bitwise operation any number of times.
        """
        MOD = 10**9 + 7

        # The constraints are nums[i] <= 10^9. 2^29 < 10^9 < 2^30.
        # So we need to consider bits from 0 to 29.
        bit_counts = [0] * 30

        # The core insight is that the operation (a, b) -> (a&b, a|b) conserves the 
        # total count of set bits at each position across the array. This allows us to
        # treat bits at each position as a pool of resources to be redistributed.
        
        # 1. Count the total number of set bits at each position.
        for num in nums:
            for j in range(30):
                if (num >> j) & 1:
                    bit_counts[j] += 1

        # 2. Construct k numbers to maximize their sum of squares.
        # To maximize a sum of squares, numbers should be as unequal as possible.
        # The greedy strategy is to build k numbers, making each as large as possible
        # with the available bits, one by one.

        result_nums = [0] * k

        # For each bit position j, we have `bit_counts[j]` bits of value 2^j.
        # We distribute these greedily to make the first number largest,
        # the second number second largest, and so on. This is achieved by
        # giving a 2^j term to the first `min(k, bit_counts[j])` numbers.
        for j in range(30):
            power_of_2 = 1 << j
            # Distribute the available bits of this power to the first numbers.
            for i in range(min(k, bit_counts[j])):
                result_nums[i] += power_of_2

        # 3. Calculate the sum of squares of these k numbers, modulo MOD.
        total_sum_sq = 0
        for num in result_nums:
            # In Python, integers have arbitrary precision, so num * num won't overflow.
            # We perform modulo at each addition to keep the cumulative sum within bounds.
            term = num * num
            total_sum_sq = (total_sum_sq + term) % MOD
            
        return total_sum_sq