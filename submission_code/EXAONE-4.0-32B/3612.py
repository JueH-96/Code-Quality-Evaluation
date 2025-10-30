class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        for a in range(0, n - 2 * k + 1):
            valid_first = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid_first = False
                    break
            
            if not valid_first:
                continue
                
            valid_second = True
            for i in range(a + k, a + 2 * k - 1):
                if nums[i] >= nums[i + 1]:
                    valid_second = False
                    break
            
            if valid_first and valid_second:
                return True
                
        return False