class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # Calculate operations required to obtain the number 0.
        # A number is special if it is divisible by 25. The number 0 is special.
        # We can obtain the number 0 by deleting all digits (n operations).
        # If the string contains at least one '0', we can obtain the number "0"
        # by keeping one '0' and deleting all other n-1 digits. This takes n-1 operations.
        # This is the best possible way to obtain the number 0.
        
        ops_to_get_zero = n # Default: delete all digits (costs n operations)
        for digit in num:
            if digit == '0':
                ops_to_get_zero = n - 1
                break # Found a '0', so n-1 operations is achievable for getting "0"
                
        # Initialize minimum operations found so far with the cost of getting 0.
        min_ops = ops_to_get_zero

        # Calculate minimum operations required to obtain a number ending in "00", "25", "50", or "75".
        # A number is divisible by 25 if its last two digits are 00, 25, 50, or 75.
        # To obtain a number ending in "XY", we need to find two digits num[i] = X and num[j] = Y with i < j
        # that become the last two digits of the resulting number.
        # To minimize deletions for a specific suffix "XY", we should find the rightmost occurrence
        # of 'Y' in num, let its index be j. Then, we find the rightmost occurrence of 'X'
        # in num at an index i < j.
        # The number of operations to make num[i] and num[j] the last two digits (by keeping num[0...i-1])
        # is the number of digits we must delete:
        # 1. All digits after index j: indices j+1, ..., n-1. Count = (n - 1) - (j + 1) + 1 = n - 1 - j.
        # 2. All digits between index i and j: indices i+1, ..., j-1. Count = (j - 1) - (i + 1) + 1 = j - 1 - i.
        # The total number of operations for this specific pair (i, j) is (n - 1 - j) + (j - 1 - i).

        targets = ["00", "25", "50", "75"]
        
        for target in targets:
            char_y = target[1]
            char_x = target[0]
            
            # Find the rightmost occurrence of char_y
            j = -1
            for k in range(n - 1, -1, -1):
                if num[k] == char_y:
                    j = k
                    break
            
            # If char_y is found, find the rightmost occurrence of char_x before index j
            if j != -1:
                i = -1
                for k in range(j - 1, -1, -1):
                    if num[k] == char_x:
                        i = k
                        break
                
                # If both char_x (at index i) and char_y (at index j) are found with i < j
                if i != -1:
                    # Calculate deletions for this pair (i, j)
                    # Deletions = (digits after j) + (digits between i and j)
                    ops = (n - 1 - j) + (j - 1 - i)
                    min_ops = min(min_ops, ops)

        return min_ops