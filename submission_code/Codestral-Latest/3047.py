class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        import math
        from collections import defaultdict

        def is_perfect_square(x):
            root = int(math.isqrt(x))
            return root * root == x

        # Dictionary to store the maximum sum for each remainder when divided by 4
        remainder_sums = defaultdict(int)

        max_sum = 0

        for num in nums:
            remainder = num % 4
            complement = (4 - remainder) % 4

            if remainder in remainder_sums:
                max_sum = max(max_sum, remainder_sums[remainder] + num)

            if complement in remainder_sums:
                max_sum = max(max_sum, remainder_sums[complement] + num)

            remainder_sums[remainder] = max(remainder_sums[remainder], num)

        return max_sum