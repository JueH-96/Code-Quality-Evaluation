class Solution:
    def minOperations(self, nums: List[int]) -> int:
        spf = [0] * (1000001)
        for i in range(2, 1000001):
            if spf[i] == 0:
                spf[i] = i
                for j in range(i * i, 1000001, i):
                    if spf[j] == 0:
                        spf[j] = i

        def get_greatest_proper_divisor(n, spf):
            if n == 1:
                return 1
            return n // spf[n]

        operations = 0
        current_nums = list(nums)
        for i in range(1, len(current_nums)):
            while current_nums[i] < current_nums[i-1]:
                greatest_proper_divisor = get_greatest_proper_divisor(current_nums[i-1], spf)
                if greatest_proper_divisor == 1:
                    return -1
                current_nums[i-1] //= greatest_proper_divisor
                operations += 1
                if current_nums[i-1] == 0:
                    return -1
        return operations