class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        mid = (n - 1) // 2
        ans = 0
        if nums[mid] == k:
            pass
        elif nums[mid] < k:
            ans += k - nums[mid]
            for i in range(mid + 1, n):
                if nums[i] < k:
                    ans += k - nums[i]
                else:
                    break
        else:
            ans += nums[mid] - k
            for i in range(mid - 1, -1, -1):
                if nums[i] > k:
                    ans += nums[i] - k
                else:
                    break
        return ans