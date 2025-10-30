class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)

        def check(freq):
            for i in range(n - freq + 1):
                subarray = nums[i : i + freq]

                # Calculate cost with median as target
                median = subarray[freq // 2]
                cost = 0
                for x in subarray:
                    cost += abs(x - median)
                if cost <= k:
                    return True

                # If freq is even, consider the other middle element as well
                if freq % 2 == 0:
                    median2 = subarray[freq // 2 - 1]
                    cost2 = 0
                    for x in subarray:
                        cost2 += abs(x - median2)
                    if cost2 <= k:
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