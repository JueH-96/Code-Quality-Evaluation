class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        ans = 0
        
        i = 0
        while i + 1 < len(nums):
            first_score = nums[i] + nums[i+1]
            count = 1
            temp_nums = nums[i+2:]
            j = 0
            while j + 1 < len(temp_nums):
                if temp_nums[j] + temp_nums[j+1] == first_score:
                    count += 1
                    j += 2
                else:
                    break
            ans = max(ans, count)
            i += 1
            
        return ans