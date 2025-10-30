class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i, num in enumerate(nums) if num == 1]
        m = len(ones)
        
        prefix_sum = [0] * (m + 1)
        for i in range(m):
            prefix_sum[i + 1] = prefix_sum[i] + ones[i]
        
        ans = float('inf')
        for i in range(m - k + 1):
            mid = (ones[i] + ones[i + k - 1]) // 2
            left = mid - ones[i]
            right = ones[i + k - 1] - mid
            changes = max(0, k - (mid - i + (i + k - 1) - mid + 1)) + max(0, k - (min(left, right) + 1))
            if changes <= maxChanges:
                ans = min(ans, (prefix_sum[i + k] - prefix_sum[mid + 1]) - (prefix_sum[mid] - prefix_sum[i]) + mid * 2 - (i + i + k - 1))
        
        return ans