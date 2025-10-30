class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        ans = float('inf')
        for j in range(l - 1, n):
            start_index = max(0, j - r + 1)
            end_index = j - l + 1
            max_val = -float('inf')
            for i in range(start_index, end_index + 1):
                if prefix[i] < prefix[j + 1] and prefix[i] > max_val:
                    max_val = prefix[i]
            if max_val != -float('inf'):
                candidate = prefix[j + 1] - max_val
                if candidate > 0:
                    if candidate == 1:
                        return 1
                    if candidate < ans:
                        ans = candidate
        return ans if ans != float('inf') else -1