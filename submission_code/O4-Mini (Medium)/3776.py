from functools import lru_cache
import sys
sys.setrecursionlimit(10000)

class Solution:
    def minCost(self, nums: list[int]) -> int:
        n = len(nums)
        
        @lru_cache(None)
        def dfs(i: int, have: int, val: int) -> int:
            """
            i: next index in nums to consider (0-based)
            have: 0 or 1, whether there's a leftover 'val' at front
            val: the value of the leftover element if have==1 (ignored if have==0)
            Returns the min cost to remove all remaining elements (including leftover if any).
            """
            # count of remaining elements
            rem = (n - i) + have
            if rem == 0:
                return 0
            # if fewer than 3 remain, remove all at once
            if rem < 3:
                # collect the remaining values
                lst = []
                if have:
                    lst.append(val)
                # append the remaining nums
                for k in range(i, n):
                    lst.append(nums[k])
                return max(lst) if lst else 0
            
            res = float('inf')
            # case: no leftover, buffer = [nums[i], nums[i+1], nums[i+2], ...]
            if have == 0:
                # remove (i, i+1), leftover nums[i+2]
                cost = max(nums[i], nums[i+1])
                res = min(res, cost + dfs(i+3, 1, nums[i+2]))
                # remove (i, i+2), leftover nums[i+1]
                cost = max(nums[i], nums[i+2])
                res = min(res, cost + dfs(i+3, 1, nums[i+1]))
                # remove (i+1, i+2), leftover nums[i]
                cost = max(nums[i+1], nums[i+2])
                res = min(res, cost + dfs(i+3, 1, nums[i]))
            else:
                # have == 1, buffer = [val, nums[i], nums[i+1], ...]
                # remove (val, nums[i]), leftover nums[i+1]
                cost = max(val, nums[i])
                res = min(res, cost + dfs(i+2, 1, nums[i+1]))
                # remove (val, nums[i+1]), leftover nums[i]
                cost = max(val, nums[i+1])
                res = min(res, cost + dfs(i+2, 1, nums[i]))
                # remove (nums[i], nums[i+1]), leftover val
                cost = max(nums[i], nums[i+1])
                res = min(res, cost + dfs(i+2, 1, val))
            
            return res
        
        # start with no leftover at position 0
        return dfs(0, 0, 0)

# Example usage:
# sol = Solution()
# print(sol.minCost([6,2,8,4]))  # outputs 12
# print(sol.minCost([2,1,3,3]))  # outputs 5