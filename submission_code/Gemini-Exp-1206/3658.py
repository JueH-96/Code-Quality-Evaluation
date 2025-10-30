class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 2:
            if nums[0] == -1 and nums[1] == -1:
                return 0
            elif nums[0] == -1 or nums[1] == -1:
                return 0
            else:
                return abs(nums[0] - nums[1])

        candidates = []
        max_diff = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i + 1] != -1:
                max_diff = max(max_diff, abs(nums[i] - nums[i + 1]))
            elif nums[i] != -1:
                candidates.append(nums[i])
            elif nums[i + 1] != -1:
                candidates.append(nums[i + 1])

        if not candidates:
            return 0

        candidates.sort()
        l, r = candidates[0], candidates[-1]

        if l == r:
            return max_diff

        ans = float('inf')
        while l <= r:
            mid = (l + r) // 2
            diff1 = max(max_diff, abs(candidates[0] - mid))
            diff2 = max(max_diff, abs(candidates[-1] - mid))
            for i in range(len(candidates) - 1):
                diff1 = max(diff1, abs(candidates[i+1] - candidates[i]))
                diff2 = max(diff2, abs(candidates[i+1] - candidates[i]))

            
            ans = min(ans, min(diff1, diff2))
            if diff1 < diff2:
                r = mid - 1
            else:
                l = mid + 1

        return ans