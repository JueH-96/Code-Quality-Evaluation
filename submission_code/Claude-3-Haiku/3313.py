class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        def strength(x):
            total = 0
            for i in range(1, x + 1):
                total += (-1) ** (i + 1) * (prefix_sum[i + x - 1] - prefix_sum[i - 1]) * (x - i + 1)
            return total
        
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            if strength(mid) >= strength(mid - 1):
                left = mid + 1
            else:
                right = mid - 1
        
        max_strength = 0
        for i in range(1, left):
            max_strength = max(max_strength, strength(i))
        
        for i in range(1, k // 2 + 1):
            max_strength = max(max_strength, strength(left - i))
        
        return max_strength