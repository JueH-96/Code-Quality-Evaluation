class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            curr_and = nums[i]
            if curr_and == k:
                count += 1
            for j in range(i+1, n):
                curr_and &= nums[j]
                if curr_and == k:
                    count += 1
                elif curr_and < k:
                    break
        
        return count