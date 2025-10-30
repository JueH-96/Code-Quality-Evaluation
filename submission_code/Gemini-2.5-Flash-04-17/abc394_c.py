import sys

# Read input string from stdin
S = sys.stdin.readline().strip()

# Use a list as a stack to build the modified string
# The stack will store characters that have been processed and are
# part of the transformed string built so far, potentially subject to further
# reduction if a 'WA' pattern is formed at the end.
result = []

# Process each character of the input string from left to right
for char in S:
    # Append the current character to the end of our result stack
    result.append(char)

    # Check if the newly appended character completes a "WA" pattern at the end.
    # If it does, replace "WA" with "AC". This replacement might itself create
    # a new "WA" pattern ending at the current position (e.g., WWA -> WAC).
    # The while loop handles these cascading reductions occurring at the end
    # of the result stack until no more "WA" patterns are found at the end.
    while len(result) >= 2 and result[-2] == 'W' and result[-1] == 'A':
        # The last two characters form the pattern "WA".
        # According to the rule, replace the leftmost "WA" with "AC".
        # The stack processes from left to right, so any "WA" entirely to the left
        # should have been reduced already. The "WA" at the end of the stack,
        # potentially formed or completed by the current character, is the leftmost
        # such pattern that can be reduced involving the current processing position.
        # Replace "WA" with "AC". This means the 'W' (result[-2]) effectively becomes 'A',
        # and the 'A' (result[-1]) effectively becomes 'C'.
        # We represent this transformation on the stack by popping the 'W' and 'A'
        # and then appending 'A' and 'C'.

        result.pop() # Remove the last character, which is 'A'
        result.pop() # Remove the second to last character, which is 'W'

        # Append the replacement characters 'A' and 'C'
        result.append('A') # The 'W' conceptually becomes 'A'
        result.append('C') # The 'A' conceptually becomes 'C'

        # The while loop will now re-check the end of the 'result' list.
        # If the newly appended 'A' forms a new 'WA' with the character now
        # preceding it (which was result[-3] originally, now result[-2]),
        # the loop condition will be true again and the reduction continues.

# After processing all characters from the input string and performing all
# necessary cascading reductions at the end of the result stack, the 'result'
# list contains the characters of the final modified string.
# Join the characters in the list into a single string for output.
print("".join(result))