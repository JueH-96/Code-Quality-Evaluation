from typing import List
import math
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # Function to find the square-free part of a positive integer
        # The square-free part of n is the product of the prime factors
        # that appear with an odd exponent in the prime factorization of n.
        # Two positive integers x and y have their product x*y be a perfect square
        # if and only if they have the same square-free part.
        # This is because x = sf(x) * a^2 and y = sf(y) * b^2.
        # x*y = sf(x) * sf(y) * (a*b)^2. For x*y to be a perfect square,
        # sf(x) * sf(y) must be a perfect square. Since sf(x) and sf(y) are
        # square-free, their product is a perfect square iff sf(x) = sf(y).
        def get_square_free(n):
            sf = 1
            i = 2
            # Iterate through potential prime factors starting from 2
            # We only need to check divisors up to sqrt(n).
            # If after dividing out all prime factors <= sqrt(n) with their
            # full powers, the remaining number > 1, this remaining number
            # must be a prime itself (greater than sqrt(original_n)).
            # The condition i * i <= n is correct because n is updated inside the loop.
            # If n becomes 1 quickly, the loop terminates quickly.
            while i * i <= n:
                if n % i == 0:
                    count = 0
                    while n % i == 0:
                        count += 1
                        n //= i
                    # If the prime factor appeared an odd number of times, include it in sf
                    if count % 2 == 1:
                        sf *= i
                i += 1

            # If after the loop, n > 1, it means the remaining n is a prime factor
            # that is greater than sqrt(original_n). This prime factor must
            # have appeared with an odd exponent in the original number.
            # So, include this remaining prime factor in sf.
            if n > 1:
                sf *= n

            return sf

        # Dictionary to store the sum of numbers for each unique square-free part
        # A complete set of numbers is formed by numbers sharing the same square-free part.
        sum_by_sf = defaultdict(int)

        # Group numbers by their square-free part and sum them up
        for num in nums:
            sf = get_square_free(num)
            sum_by_sf[sf] += num

        # The maximum sum among all groups is the maximum element-sum
        # of a complete subset of indices.
        # A complete subset corresponds to indices whose elements form a complete set of numbers.
        # We sum up all numbers sharing the same square-free part to maximize the sum
        # for that group.
        # The overall maximum sum is the maximum among these group sums.
        # Since the constraints state 1 <= n <= 10^4, the input list nums is not empty.
        # The sum_by_sf dictionary will therefore have at least one entry.
        # Each single-element subset {nums[i]} is always complete because nums[i]*nums[i]
        # is a perfect square. The sum is nums[i]. The logic covers this: if a square-free
        # part corresponds to only one number, its sum is just that number.
        # If multiple numbers share the same square-free part, their combined sum might be
        # greater than any single number.
        return max(sum_by_sf.values())