class Solution:
    def minCost(self, nums: List[int]) -> int:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(arr):
            if not arr:
                return 0
            n = len(arr)
            if n == 1:
                return arr[0]
            if n == 2:
                return max(arr[0], arr[1])
            # Option 1: remove first two elements
            option1 = max(arr[0], arr[1]) + dp(tuple(arr[2:]))
            # Option 2: remove first and third elements
            option2 = max(arr[0], arr[2]) + dp(tuple(arr[1:3] + arr[3:]))
            # Option 3: remove second and third elements
            option3 = max(arr[1], arr[2]) + dp(tuple([arr[0]] + arr[3:]))
            return min(option1, option2, option3)
        
        return dp(tuple(nums))