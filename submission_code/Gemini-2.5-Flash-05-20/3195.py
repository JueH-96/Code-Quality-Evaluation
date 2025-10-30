class Solution:
    def minimumSteps(self, s: str) -> int:
        # Initialize total_steps to count the minimum number of swaps.
        # This will accumulate the count of inversions ('1' appearing before '0').
        total_steps = 0
        
        # Initialize ones_count to keep track of the number of '1's encountered so far
        # from the beginning of the string.
        ones_count = 0
        
        # Iterate through each character in the input string s.
        for char in s:
            if char == '1':
                # If the current character is '1' (black ball),
                # increment ones_count. This '1' will contribute to the
                # steps for any '0's that appear after it.
                ones_count += 1
            else: # char == '0'
                # If the current character is '0' (white ball),
                # it means this '0' is currently to the right of 'ones_count'
                # number of '1's that we have already processed.
                # Each of these 'ones_count' '1's forms an inversion with
                # the current '0'. To move this '0' to the left of all those '1's,
                # or those '1's to the right of this '0', we need 'ones_count'
                # adjacent swaps involving this '0'.
                # We add this count to our total_steps.
                total_steps += ones_count
        
        # The total_steps accumulated represents the minimum number of adjacent swaps
        # required to group all '1's to the right and all '0's to the left.
        return total_steps