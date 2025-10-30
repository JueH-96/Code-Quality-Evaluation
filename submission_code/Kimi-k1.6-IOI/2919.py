class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        n = len(usageLimits)
        prefix = [0] * n
        prefix[0] = usageLimits[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + usageLimits[i]
        
        low, high = 1, n
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            possible = True
            for m in range(1, mid + 1):
                required = m * (m + 1) // 2
                if prefix[m-1] < required:
                    possible = False
                    break
            if possible:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans