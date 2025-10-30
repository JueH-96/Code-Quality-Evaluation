class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        longest_seq_sum = nums[0]
        current_seq_sum = nums[0]
        prev_num = nums[0]
        
        for num in nums[1:]:
            if num == prev_num + 1:
                current_seq_sum += num
            else:
                if current_seq_sum > longest_seq_sum:
                    longest_seq_sum = current_seq_sum
                current_seq_sum = num
            prev_num = num
        
        if current_seq_sum > longest_seq_sum:
            longest_seq_sum = current_seq_sum
        
        x = longest_seq_sum
        while x in nums:
            x += 1
        return x