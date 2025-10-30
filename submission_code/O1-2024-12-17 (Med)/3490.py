class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # 1) Count how many even and how many odd elements there are
        even_count = sum(1 for x in nums if x % 2 == 0)
        odd_count = len(nums) - even_count
        
        # Longest subsequence where consecutive sums are even
        # (i.e., all elements in the subsequence have the same parity)
        longest_same_parity = max(even_count, odd_count)
        
        # 2) Find the longest subsequence where consecutive sums are odd
        # (i.e., elements strictly alternate in parity)
        longest_alternating = 1  # at least one element can be taken
        last_parity = nums[0] % 2
        
        for i in range(1, len(nums)):
            current_parity = nums[i] % 2
            if current_parity != last_parity:
                longest_alternating += 1
                last_parity = current_parity
        
        # The answer is the maximum of these two possible subsequences
        return max(longest_same_parity, longest_alternating)