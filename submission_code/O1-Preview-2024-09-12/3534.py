class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import defaultdict

        swapped_numbers = {}
        for num in set(nums):
            digits = list(str(num))
            n_digits = len(digits)
            swaps = set()
            for i in range(n_digits):
                for j in range(i, n_digits):
                    swapped = digits.copy()
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    swapped_num = int(''.join(swapped))
                    swaps.add(swapped_num)
            swapped_numbers[num] = swaps

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                num_i = nums[i]
                num_j = nums[j]
                if num_j in swapped_numbers[num_i] or num_i in swapped_numbers[num_j]:
                    count += 1

        return count