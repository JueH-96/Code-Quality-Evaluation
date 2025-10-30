import sys

def solve():
    S = sys.stdin.readline().strip()
    
    # Use a list as a stack to build the resulting string
    stack = []
    
    # Process the input string character by character
    for char in S:
        # Append the current character to the stack
        stack.append(char)
        
        # Check if the last two characters on the stack form the pattern "WA"
        # If they do, replace them with "AC" and repeat the check,
        # as this replacement might create a new "WA" pattern ending at the current position.
        # The while loop handles this cascading effect at the end of the stack.
        while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
            # Found "WA" at the end of the stack (stack[-2:]).
            # Replace this "WA" with "AC".
            # This is achieved by removing the last two characters ('W' and 'A')
            # and then adding 'A' and 'C'.
            
            # The 'A' from the "WA" is removed.
            stack.pop() 
            # The 'W' from the "WA" is removed.
            stack.pop() 
            
            # The 'W' that was removed is replaced by 'A'. Append this 'A'.
            stack.append('A')
            # The original 'A' that was removed is replaced by 'C'. Append this 'C'.
            stack.append('C')
            
            # After appending the new 'A' and 'C', the loop condition is checked again.
            # This correctly handles cases where the new 'A' (now at stack[-2])
            # forms a "WA" with the character now at stack[-3] (which was before the original 'W').

    # Join the characters remaining in the stack to form the final string
    print("".join(stack))

solve()