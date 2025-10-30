class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def get_greatest_proper_divisor(n):
            if n <= 1:
                return 1  # Handle cases where n is 1 or less
            i = 2
            while i * i <= n:
                if n % i == 0:
                    return n // i
                i += 1
            return 1

        operations = 0
        n = len(nums)
        for i in range(n - 2, -1, -1):
            if nums[i] > nums[i + 1]:
                original_num = nums[i]
                while nums[i] > nums[i + 1]:
                    divisor = get_greatest_proper_divisor(nums[i])
                    if divisor == 1 and nums[i] > nums[i+1]: #Handle case where no proper divisor exists and num[i] > num[i+1]
                        operations = -1
                        break
                    nums[i] //= divisor
                    operations += 1
                if operations == -1:
                    break

        return operations