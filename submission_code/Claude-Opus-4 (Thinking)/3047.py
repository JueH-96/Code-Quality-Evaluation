class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free_part(n):
            result = 1
            i = 2
            while i * i <= n:
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                if count % 2 == 1:
                    result *= i
                i += 1
            if n > 1:
                result *= n
            return result
        
        n = len(nums)
        group_sums = {}
        
        for i in range(1, n + 1):
            sfp = square_free_part(i)
            if sfp not in group_sums:
                group_sums[sfp] = 0
            group_sums[sfp] += nums[i - 1]
        
        return max(group_sums.values())