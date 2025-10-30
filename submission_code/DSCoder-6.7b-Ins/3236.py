class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        nums.sort()
        longest_prefix_sum = 0
        longest_prefix_length = 0
        current_prefix_sum = 0
        current_prefix_length = 0
        for i in range(len(nums)):
            if i == 0 or nums[i] == nums[i-1] + 1:
                current_prefix_sum += nums[i]
                current_prefix_length += 1
                if current_prefix_length > longest_prefix_length:
                    longest_prefix_sum = current_prefix_sum
                    longest_prefix_length = current_prefix_length
            else:
                current_prefix_sum = nums[i]
                current_prefix_length = 1
        return longest_prefix_sum + 1 if longest_prefix_sum + 1 > nums[-1] else nums[-1] + 1