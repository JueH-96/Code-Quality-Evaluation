import bisect

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        def is_palindrome(x):
            s = str(x)
            return s == s[::-1]
        
        candidates = []
        if n % 2 == 1:
            m = nums[n // 2]
            start = max(1, m - 200)
            end = min(10**9 - 1, m + 200)
            for num in range(start, end + 1):
                if is_palindrome(num):
                    candidates.append(num)
        else:
            s = nums[(n//2)-1]
            t = nums[n//2]
            # Generate candidates around s
            start_s = max(1, s - 200)
            end_s = min(10**9 - 1, s + 200)
            for num in range(start_s, end_s + 1):
                if is_palindrome(num):
                    candidates.append(num)
            # Generate candidates around t
            start_t = max(1, t - 200)
            end_t = min(10**9 - 1, t + 200)
            for num in range(start_t, end_t + 1):
                if is_palindrome(num):
                    candidates.append(num)
        
        # Remove duplicates
        candidates = list(set(candidates))
        if not candidates:
            return 0
        
        min_cost = float('inf')
        for y in candidates:
            idx = bisect.bisect_right(nums, y)
            left = y * idx - prefix[idx]
            right = (prefix[-1] - prefix[idx]) - y * (n - idx)
            total = left + right
            if total < min_cost:
                min_cost = total
        return min_cost