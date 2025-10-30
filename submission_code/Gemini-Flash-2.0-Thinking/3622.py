class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)

        def check(freq):
            for i in range(n - freq + 1):
                subarray = nums[i : i + freq]
                target = subarray[-1]
                cost = sum(target - x for x in subarray[:-1])
                if cost <= numOperations:
                    return True
            return False

        left, right = 1, n
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans