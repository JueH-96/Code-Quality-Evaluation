class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]
        
        min_diff = float('inf')
        for i in range(n + 1):
            left, right = i, n
            while left < right:
                mid = (left + right) // 2
                or_value = prefix_or[mid] ^ prefix_or[i]
                diff = abs(k - or_value)
                min_diff = min(min_diff, diff)
                if or_value < k:
                    left = mid + 1
                else:
                    right = mid
        
        return min_diff