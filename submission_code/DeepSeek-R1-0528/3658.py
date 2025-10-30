class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        INF = 10**18
        known_max = 0
        for num in nums:
            if num != -1:
                if num > known_max:
                    known_max = num
        
        def can(d):
            for i in range(n - 1):
                if nums[i] != -1 and nums[i+1] != -1:
                    if abs(nums[i] - nums[i+1]) > d:
                        return False
            
            ranges = []
            for i in range(n):
                if nums[i] != -1:
                    continue
                low = -INF
                high = INF
                if i - 1 >= 0 and nums[i-1] != -1:
                    low = max(low, nums[i-1] - d)
                    high = min(high, nums[i-1] + d)
                if i + 1 < n and nums[i+1] != -1:
                    low = max(low, nums[i+1] - d)
                    high = min(high, nums[i+1] + d)
                if low > high:
                    return False
                ranges.append((low, high))
            
            if not ranges:
                return True
            
            candidates = []
            ranges_sorted_by_right = sorted(ranges, key=lambda x: x[1])
            candidates.append(ranges_sorted_by_right[0][1])
            ranges_sorted_by_left = sorted(ranges, key=lambda x: x[0])
            candidates.append(ranges_sorted_by_left[-1][0])
            
            for x in candidates:
                not_contain = []
                for interval in ranges:
                    l, r = interval
                    if x < l or x > r:
                        not_contain.append(interval)
                if not not_contain:
                    return True
                L_val = max(interv[0] for interv in not_contain)
                R_val = min(interv[1] for interv in not_contain)
                if L_val > R_val:
                    continue
                if max(L_val, x - d) <= min(R_val, x + d):
                    return True
            return False
        
        lo = 0
        hi = known_max
        ans = 10**18
        while lo <= hi:
            mid = (lo + hi) // 2
            if can(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans