class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def get_greatest_proper_divisor(n):
            if n == 1:
                return 1
            for i in range(n // 2, 0, -1):
                if n % i == 0:
                    return i
            return 1

        def get_min_val(n):
            while True:
                divisor = get_greatest_proper_divisor(n)
                if divisor == 1:
                    return n
                n //= divisor

        n = len(nums)
        operations = 0
        min_vals = [get_min_val(num) for num in nums]

        if any(min_vals[i] > min_vals[i+1] for i in range(n-1)):
            return -1
        
        for i in range(n - 2, -1, -1):
            while nums[i] > nums[i + 1] and nums[i] > 1:
                divisor = get_greatest_proper_divisor(nums[i])
                nums[i] //= divisor
                operations += 1
            if nums[i] > nums[i+1]:
                return -1

        return operations