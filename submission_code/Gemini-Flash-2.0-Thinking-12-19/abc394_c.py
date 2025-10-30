import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# Use a list as a stack for efficient appends and pops from the end.
# The stack will store characters of the string being built, after applying
# the "WA" -> "AC" replacements greedily from left to right.
stack = []

# Process the string character by character
for char in S:
    # Always append the current character to the stack first.
    # The character is added to the end of the string being built.
    stack.append(char)
    
    # After appending the character, check if the last two characters now form "WA".
    # If they do, replace the "WA" with "AC" on the stack.
    # This check and replacement process must be repeated as long as the end
    # of the stack is "WA", because replacing "WA" with "AC" might create a
    # new "WA" with the character immediately preceding the original "WA".
    # Example: ...XWWA -> ...XWW(A) -> append A -> ...XWWA
    # stack[-2:] is WA (the second W and the A). Replace it.
    # Pop A, Pop W -> ...XW. Push A, Push C -> ...XWAC.
    # Now the end is AC. The while loop checks stack[-2:] which is AC, not WA.
    # However, the character before the original WA (the first W) is now
    # adjacent to the new A. This combination (...X W A...) forms a new WA
    # if X is empty or not W. If X was W, it forms ...W A C. The WA is now
    # stack[-3]stack[-2]. The standard stack method with the while loop
    # correctly handles this cascade because the replacement "WA"->"AC" means
    # the 'W' is replaced by 'A' and 'A' by 'C'. The character immediately
    # preceding the original 'W' now becomes adjacent to the new 'A', which
    # is exactly `stack[-2]` after the pop(2) and push('A') steps.
    
    # The condition checks if the stack has at least two elements and if the
    # last element is 'A' and the second to last is 'W'.
    while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
        # Found 'WA' at the end of the stack.
        # Replace 'WA' with 'AC' on the stack.
        # This involves removing 'W' and 'A' and then adding 'A' and 'C'.
        
        # Remove 'A' (the last character)
        stack.pop()
        # Remove 'W' (the second to last character)
        stack.pop()
        
        # Append 'A' (the first character of the replacement "AC")
        stack.append('A')
        # Append 'C' (the second character of the replacement "AC")
        stack.append('C')
        
        # The while loop condition will be checked again with the potentially
        # modified end of the stack. This handles cascading replacements.

# After processing all characters in the input string, the stack
# contains the characters of the final resulting string.
result = "".join(stack)

# Print the resulting string to standard output
print(result)