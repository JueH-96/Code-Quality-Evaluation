class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        @functools.lru_cache(None)
        def dfs(left: int, right: int) -> bool:
            if right - left + 1 <= 2:
                return True
            
            total = sum(nums[left:right+1])
            
            for i in range(left, right):
                left_sum = sum(nums[left:i+1])
                right_sum = total - left_sum
                
                if (left_sum >= m or i - left + 1 == 1) and (right_sum >= m or right - i == 1):
                    if dfs(left, i) and dfs(i+1, right):
                        return True
            
            return False
        
        return dfs(0, n-1)