class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        low = 0
        high = max(nums)
        ans = high
        while low <= high:
            mid = (low + high) // 2
            count = 1
            curr = nums[0]
            for i in range(1, n):
                if (curr | nums[i]) > mid:
                    count += 1
                    curr = nums[i]
                else:
                    curr |= nums[i]
            if count <= k + 1:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans