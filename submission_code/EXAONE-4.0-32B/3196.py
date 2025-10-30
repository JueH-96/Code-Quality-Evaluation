class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        low, high = 1, n
        ans = 0
        while low <= high:
            L = (low + high) // 2
            found = False
            for i in range(0, n - L + 1):
                j = i + L - 1
                mid = i + (L - 1) // 2
                left_ops = nums[mid] * (mid - i) - (prefix[mid] - prefix[i])
                right_ops = (prefix[j + 1] - prefix[mid + 1]) - nums[mid] * (j - mid)
                total_ops = left_ops + right_ops
                if total_ops <= k:
                    found = True
                    break
            if found:
                ans = L
                low = L + 1
            else:
                high = L - 1
        return ans