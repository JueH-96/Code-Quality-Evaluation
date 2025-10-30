class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        max_sum = 0
        current_sum = 0
        longest_seq_len = 0
        
        if not nums:
            return 1

        for i in range(len(nums)):
            if i == 0:
                current_sum = nums[i]
                longest_seq_len = 1
                max_sum = current_sum
            else:
                if nums[i] == nums[i-1] + 1:
                    current_sum += nums[i]
                    longest_seq_len += 1
                else:
                    current_sum = nums[i]
                    longest_seq_len = 1
                
            max_sum = max(max_sum, current_sum)

        x = max_sum
        while x in nums:
            x += 1
        return x