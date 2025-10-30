class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        # prefixOK[i] = True if nums[0..i] is strictly increasing
        prefixOK = [False] * n
        for i in range(n):
            if i == 0:
                prefixOK[i] = True
            else:
                prefixOK[i] = prefixOK[i - 1] and nums[i - 1] < nums[i]
        # suffixOK[i] = True if nums[i..n-1] is strictly increasing
        suffixOK = [False] * n
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                suffixOK[i] = True
            else:
                suffixOK[i] = suffixOK[i + 1] and nums[i] < nums[i + 1]
        count = 0
        # try removing every non-empty subarray nums[l..r]
        for l in range(n):
            for r in range(l, n):
                # prefix part must be strictly increasing
                if l > 0 and not prefixOK[l - 1]:
                    continue
                # suffix part must be strictly increasing
                if r + 1 < n and not suffixOK[r + 1]:
                    continue
                # if both parts exist, they must join strictly
                if l > 0 and r + 1 < n and nums[l - 1] >= nums[r + 1]:
                    continue
                count += 1
        return count