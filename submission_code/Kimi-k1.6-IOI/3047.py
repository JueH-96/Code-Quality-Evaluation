class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free_part(i):
            if i == 1:
                return 1
            s = 1
            remaining = i
            p = 2
            while p * p <= remaining:
                if remaining % p == 0:
                    exponent = 0
                    while remaining % p == 0:
                        exponent += 1
                        remaining //= p
                    if exponent % 2 == 1:
                        s *= p
                p += 1
            if remaining > 1:
                s *= remaining
            return s
        
        sum_groups = {}
        max_sum = 0
        for idx in range(1, len(nums) + 1):
            sfp = square_free_part(idx)
            if sfp in sum_groups:
                sum_groups[sfp] += nums[idx - 1]
            else:
                sum_groups[sfp] = nums[idx - 1]
            if sum_groups[sfp] > max_sum:
                max_sum = sum_groups[sfp]
        
        return max_sum