class Solution:
    def minCost(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def solve(arr_tuple):
            if len(arr_tuple) == 0:
                return 0
            if len(arr_tuple) <= 2:
                return max(arr_tuple)
            
            # Try all possible pairs from first 3 elements
            option1 = max(arr_tuple[0], arr_tuple[1]) + solve(arr_tuple[2:])
            option2 = max(arr_tuple[0], arr_tuple[2]) + solve((arr_tuple[1],) + arr_tuple[3:])
            option3 = max(arr_tuple[1], arr_tuple[2]) + solve((arr_tuple[0],) + arr_tuple[3:])
            
            return min(option1, option2, option3)
        
        return solve(tuple(nums))