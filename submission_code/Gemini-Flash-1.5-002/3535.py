class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        dp = {}

        def count_monotonic_pairs(index, arr1_last, arr2_last):
            if index == n:
                return 1
            if (index, arr1_last, arr2_last) in dp:
                return dp[(index, arr1_last, arr2_last)]

            count = 0
            for arr1_val in range(arr1_last, nums[index] + 1):
                arr2_val = nums[index] - arr1_val
                if arr2_val <= arr2_last:
                    count = (count + count_monotonic_pairs(index + 1, arr1_val, arr2_val)) % mod

            dp[(index, arr1_last, arr2_last)] = count
            return count

        return count_monotonic_pairs(0, 0, 51)