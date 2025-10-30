class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        prev_sum = 0
        curr_sum = 0
        count = 0
        
        for num in nums:
            curr_sum += num
            if curr_sum >= prev_sum:
                count += 1
                prev_sum = curr_sum
                curr_sum = 0
        
        return count