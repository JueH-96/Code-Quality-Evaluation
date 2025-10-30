from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # total distinct values in the entire array
        target_distinct = len(set(nums))
        
        freq = {}
        distinct = 0
        res = 0
        right = 0
        
        # for each start index 'left', find the minimal 'right' so that
        # nums[left:right] (inclusive of right-1) contains all distinct values
        for left in range(n):
            # expand right until we have all distinct values or run out
            while right < n and distinct < target_distinct:
                x = nums[right]
                freq[x] = freq.get(x, 0) + 1
                if freq[x] == 1:
                    distinct += 1
                right += 1
            
            # if we never reached the full distinct count, no further left
            # will work either (right is at end), so we break
            if distinct < target_distinct:
                break
            
            # all subarrays starting at 'left' and ending at any index
            # from right-1 up to n-1 are complete
            res += (n - right + 1)
            
            # before moving left forward, remove its contribution
            y = nums[left]
            freq[y] -= 1
            if freq[y] == 0:
                distinct -= 1
        
        return res