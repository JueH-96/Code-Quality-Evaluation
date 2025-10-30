class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def greatest_proper_divisor(n):
            if n == 1:
                return 1
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return n // i
            return 1

        n = len(nums)
        operations = 0
        for i in range(1, n):
            if nums[i] >= nums[i - 1]:
                continue
            
            curr = nums[i]
            ops = 0
            while curr < nums[i-1]:
                if curr == 1:
                    return -1
                curr = curr // greatest_proper_divisor(curr)
                ops += 1
            nums[i] = curr
            operations += ops
        return operations