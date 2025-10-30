class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # 1. Find the length of the longest sequential prefix
        n = len(nums)
        longest_seq_len = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                longest_seq_len += 1
            else:
                break
        
        # 2. Calculate the sum of that longest sequential prefix
        prefix_sum = sum(nums[:longest_seq_len])
        
        # 3. Find and return the smallest missing integer >= prefix_sum
        existing_nums = set(nums)
        candidate = prefix_sum
        while True:
            if candidate not in existing_nums:
                return candidate
            candidate += 1