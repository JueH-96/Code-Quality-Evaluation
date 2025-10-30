class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort(reverse=True)
        n = len(usageLimits)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + usageLimits[i]
        total = prefix[-1]
        
        low = 0
        high = min(n, int((2 * total) ** 0.5) + 1)
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            required_sum = mid * (mid + 1) // 2
            if total < required_sum:
                high = mid - 1
                continue
            
            valid = True
            for k in range(1, mid + 1):
                required = k * (k + 1) // 2
                if prefix[k] < required:
                    valid = False
                    break
            
            if valid:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans