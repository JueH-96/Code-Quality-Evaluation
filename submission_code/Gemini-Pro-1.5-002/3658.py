class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        missing = []
        for i in range(n):
            if nums[i] == -1:
                missing.append(i)
        
        if not missing:
            max_diff = 0
            for i in range(n - 1):
                max_diff = max(max_diff, abs(nums[i] - nums[i+1]))
            return max_diff
        
        def check(diff):
            for x in range(1, 101):
                y = x + diff
                temp = list(nums)
                valid = True
                for i in missing:
                    prev = temp[i-1] if i > 0 else -float('inf')
                    nxt = temp[i+1] if i < n - 1 else float('inf')

                    if abs(x - prev) <= diff and abs(x - nxt) <= diff:
                        temp[i] = x
                    elif abs(y - prev) <= diff and abs(y - nxt) <= diff:
                        temp[i] = y
                    else:
                        valid = False
                        break
                
                if valid:
                    return True
            return False

        left, right = 0, 2 * 10**9
        ans = right
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans