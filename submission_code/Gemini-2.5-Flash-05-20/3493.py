class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        # current_ones represents the count of '1's that are currently to the left
        # of the iterated '0's and can potentially perform an operation.
        # When a '0' is encountered, all '1's accumulated in 'current_ones'
        # are considered to have performed their move past this block of '0's.
        current_ones = 0

        for char in s:
            if char == '1':
                # Encountered a '1'. It becomes available for operations.
                current_ones += 1
            else:  # char == '0'
                # Encountered a '0'. If there are '1's to its left that can move.
                if current_ones > 0:
                    # Each of the 'current_ones' contributes to operations
                    # as they conceptually move past this '0' (and any subsequent '0's
                    # in the same block of contiguous '0's).
                    # This logic aligns with the problem statement that a '1' moves
                    # past multiple '0's as a single operation (e.g., '100' -> '001' is 1 op).
                    # However, when distinct '1's can move past the same '0' (due to '0's shifting left),
                    # this method accumulates those operations.
                    # Example: "110" -> 2 ops.
                    #   '1': current_ones = 1
                    #   '1': current_ones = 2
                    #   '0': operations += 2 (current_ones), operations = 2.
                    # This is correct.
                    operations += current_ones
                    
                    # After the 'current_ones' have conceptually moved past this '0'-block,
                    # they are considered to be to the right of it. They are no longer
                    # active for this particular block of '0's. So, reset current_ones.
                    current_ones = 0
        
        return operations