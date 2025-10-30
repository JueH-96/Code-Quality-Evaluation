from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Case 1: Valid subsequences where (sub[i] + sub[i+1]) % 2 == 0 for all i.
        # This implies all elements in the subsequence must have the same parity.
        # We count the total number of even and odd numbers in the original array.
        count_even = 0
        count_odd = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
            else:
                count_odd += 1
        
        # The longest subsequence for this case is the maximum of the even count or odd count.
        max_len_same_parity = max(count_even, count_odd)

        # Case 2: Valid subsequences where (sub[i] + sub[i+1]) % 2 == 1 for all i.
        # This implies that elements in the subsequence must strictly alternate parities.
        # For example, [odd, even, odd, ...] or [even, odd, even, ...].
        # We find the length of the longest such subsequence using a greedy approach.
        
        current_len_alternating = 0
        # prev_parity stores the parity of the last element chosen for the alternating subsequence.
        # Initialize to -1 to ensure the first element (0 or 1 parity) is always picked.
        prev_parity = -1 

        for num in nums:
            current_parity = num % 2
            
            # If this is the first element being considered for an alternating subsequence,
            # or if its parity is different from the last element chosen,
            # we can extend our current alternating subsequence.
            if current_len_alternating == 0 or current_parity != prev_parity:
                current_len_alternating += 1
                prev_parity = current_parity
            # If current_parity == prev_parity, this 'num' cannot extend the *current* 
            # alternating sequence. Since we are looking for a subsequence, we simply skip it.
            # The 'current_len_alternating' variable correctly tracks the maximum length 
            # by greedily selecting only the alternating elements.

        # After iterating through all numbers, current_len_alternating holds the length
        # of the longest alternating subsequence found.
        max_len_alternating = current_len_alternating
        
        # The overall longest valid subsequence is the maximum of the lengths found in the two cases.
        return max(max_len_same_parity, max_len_alternating)