from bisect import bisect_left, bisect_right

class Solution:
    def maxFrequency(self, nums, k, numOperations):
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        # Generate all possible candidate targets
        candidates = set()
        for num in nums:
            candidates.add(num)
            candidates.add(num - k)
            candidates.add(num + k)
        # Ensure edge cases are covered
        if n > 0:
            candidates.add(nums[0] + k)
            candidates.add(nums[-1] - k)
        
        max_freq = 0
        for target in candidates:
            low = target - k
            high = target + k
            # Find left and right bounds using binary search
            left = bisect_left(nums, low)
            if left >= n:
                S_total = 0
            else:
                right = bisect_right(nums, high) - 1
                if left > right:
                    S_total = 0
                else:
                    S_total = right - left + 1
            # Find the count of elements equal to target (A)
            a_left = bisect_left(nums, target)
            if a_left >= n or nums[a_left] != target:
                A = 0
            else:
                a_right = bisect_right(nums, target)
                A = a_right - a_left
            # Compute candidate frequency
            current = min(S_total, A + numOperations)
            if current > max_freq:
                max_freq = current
        return max_freq