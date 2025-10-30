class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count how many even and how many odd numbers in nums
        even_count = 0
        odd_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        
        # Case 1: Subsequence where every pair-sum is even
        #         => all elements have the same parity (all even or all odd)
        same_parity_max = max(even_count, odd_count)
        
        # Case 2: Subsequence where every pair-sum is odd
        #         => adjacent elements must alternate in parity
        # We can find the longest such subsequence by a single pass:
        alt_length = 1  # At least one element can be taken
        current_parity = nums[0] % 2
        for i in range(1, len(nums)):
            if (nums[i] % 2) != current_parity:
                alt_length += 1
                current_parity = 1 - current_parity  # Flip parity
                
        # The answer is the maximum of the two subsequence lengths
        return max(same_parity_max, alt_length)