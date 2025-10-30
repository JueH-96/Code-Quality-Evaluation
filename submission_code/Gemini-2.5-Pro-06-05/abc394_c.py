# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string from standard input, applies the specified transformation,
    and prints the resulting string to standard output.
    """
    # Read the input string. Using sys.stdin.readline for efficiency with large inputs.
    S = sys.stdin.readline().strip()

    # res_parts will store parts of the resulting string to be joined at the end.
    # This is more efficient than repeated string concatenation.
    res_parts = []
    
    # w_count tracks the number of consecutive 'W's we have seen.
    w_count = 0

    for char in S:
        if char == 'W':
            w_count += 1
        else:
            # A non-'W' character is encountered. This terminates any ongoing
            # sequence of 'W's. We need to process the accumulated 'W's.
            if w_count > 0:
                if char == 'A':
                    # A block of 'W's is followed by 'A'.
                    # This triggers the transformation W^k A -> AC^k.
                    res_parts.append('A')
                    res_parts.append('C' * w_count)
                else:
                    # A block of 'W's is followed by another character (e.g., 'B', 'C').
                    # The 'W's are not transformed. Append them as they are,
                    # followed by the current character.
                    res_parts.append('W' * w_count)
                    res_parts.append(char)
                
                # The block of 'W's has been processed, so reset the counter.
                w_count = 0
            else:
                # The current character is not 'W' and there's no preceding 'W' block.
                # Simply append the character to the result.
                res_parts.append(char)

    # If the input string ends with a sequence of 'W's, they need to be appended
    # as they were not processed in the loop (which triggers on non-'W' chars).
    if w_count > 0:
        res_parts.append('W' * w_count)

    # Join all the parts and print the final string.
    print("".join(res_parts))

solve()