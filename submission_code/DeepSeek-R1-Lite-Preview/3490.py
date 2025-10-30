class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the number of even and odd elements
        count_even = sum(1 for num in nums if num % 2 == 0)
        count_odd = len(nums) - count_even
        
        # Find the longest alternating parity subsequence
        if not nums:
            return 0
        
        last_parity = nums[0] % 2
        current_length = 1
        for num in nums[1:]:
            current_parity = num % 2
            if current_parity != last_parity:
                current_length += 1
                last_parity = current_parity
        
        # The maximum between same parity subsequence and alternating parity subsequence
        return max(max(count_even, count_odd), current_length)