class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        def is_possible(frequency):
            for i in range(frequency - 1, n):
                operations_needed = 0
                for j in range(i - frequency + 1, i):
                    operations_needed += nums[i] - nums[j]
                if operations_needed <= numOperations:
                    return True
            return False

        low = 1
        high = n
        ans = 1
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans