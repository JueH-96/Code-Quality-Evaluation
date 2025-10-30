class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # 1) Find the longest sequential prefix
        #    The prefix is sequential if nums[j] = nums[j-1] + 1 for all j > 0.
        #    We break as soon as the sequence is not consecutive.
        longest_prefix_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                longest_prefix_len += 1
            else:
                break
        
        # 2) Compute the sum of this longest sequential prefix
        prefix_sum = sum(nums[:longest_prefix_len])
        
        # 3) Starting from prefix_sum, find the smallest integer x not in nums
        #    such that x >= prefix_sum.
        existing_nums = set(nums)
        candidate = prefix_sum
        while True:
            if candidate not in existing_nums:
                return candidate
            candidate += 1