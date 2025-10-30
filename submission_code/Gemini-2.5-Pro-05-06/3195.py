class Solution:
    def minimumSteps(self, s: str) -> int:
        # total_swaps will accumulate the minimum number of steps.
        # Type hint 'int' is for clarity; Python handles large integers automatically.
        total_swaps: int = 0
        
        # ones_count tracks the number of '1's encountered so far
        # while iterating from left to right. These are the '1's that
        # are to the left of the current character being processed.
        ones_count: int = 0
        
        # Iterate through the string character by character.
        for char_val in s:
            if char_val == '1':
                # If the current character is a '1' (black ball),
                # increment the count of '1's seen.
                # This '1' itself doesn't cause swaps at this moment,
                # but it will need to be moved past if a '0' is encountered later.
                ones_count += 1
            else:  # char_val == '0'
                # If the current character is a '0' (white ball),
                # it means this '0' is currently to the right of 'ones_count' '1's.
                # In the target configuration ("0...01...1"), this '0' must be
                # to the left of all these '1's.
                # To achieve this, this '0' must effectively be swapped past
                # all 'ones_count' '1's that are to its left.
                # Each such "pass" costs one step.
                total_swaps += ones_count
                
        return total_swaps