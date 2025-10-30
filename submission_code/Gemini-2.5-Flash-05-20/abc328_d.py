import sys

def solve():
    S = sys.stdin.readline().strip()

    stack = []
    
    for char in S:
        stack.append(char)
        
        # Check if the last three characters in the stack form "ABC"
        # We need at least 3 characters to form "ABC"
        if len(stack) >= 3:
            # Check the last three elements explicitly
            if stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
                # If "ABC" is found, remove the last three characters
                stack.pop() # Remove 'C'
                stack.pop() # Remove 'B'
                stack.pop() # Remove 'A'
    
    # Join the remaining characters in the stack to form the final string
    # and print it to standard output.
    sys.stdout.write("".join(stack) + "
")

# Call the solve function to run the program
solve()