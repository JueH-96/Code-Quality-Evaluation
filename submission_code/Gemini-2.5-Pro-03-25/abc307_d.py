import sys

# Function to solve the problem
def solve():
    """
    Reads the input string S and processes it according to the problem rules,
    then prints the final resulting string to standard output.
    """
    # Read N (length of the string, not strictly needed for the algorithm but part of input)
    # Using sys.stdin.readline for potentially large inputs.
    n = int(sys.stdin.readline()) 
    # Read the string S
    s = sys.stdin.readline().strip()

    # Stack to store the characters of the resulting string being built.
    # We use a Python list as a stack.
    stack = []
    
    # Stack to store the indices of '(' characters within the `stack` list.
    # When a '(' is pushed onto `stack`, its index in `stack` is pushed onto `paren_indices_stack`.
    # This helps identify the starting position of a substring to remove when a matching ')' is found.
    paren_indices_stack = []

    # Iterate through each character of the input string S
    for char in s:
        if char == '(':
            # If the character is an opening parenthesis '(':
            # Store the current length of the stack. This value represents the index 
            # where the '(' will be placed in the `stack`.
            paren_indices_stack.append(len(stack))
            # Push the '(' character onto the main stack.
            stack.append(char)
        elif char == ')':
            # If the character is a closing parenthesis ')':
            # Check if there is a matching opening parenthesis available on the `paren_indices_stack`.
            if paren_indices_stack:
                # If yes, there is a matching '('. This signifies a pair `(...)` that needs to be removed.
                # The content inside this pair consists of characters that either are letters 
                # or belong to nested `(...)` pairs that have already been processed and removed.
                
                # Pop the index of the most recent '(' from `paren_indices_stack`.
                # This index marks the start of the substring to be removed in the `stack`.
                start_index = paren_indices_stack.pop()
                
                # Remove characters from the `stack` starting from the position `start_index` up to the current end.
                # This effectively removes the matched pair `(...)` including the parentheses and its content.
                # We achieve this efficiently by popping elements from the end of the `stack` list 
                # until its length becomes equal to `start_index`. This operation has an amortized O(1) cost per pop.
                # Overall, each character is pushed and popped at most once, leading to O(N) total time complexity.
                while len(stack) > start_index:
                    stack.pop()
            else:
                # If `paren_indices_stack` is empty, this ')' does not have a matching '('.
                # It's an unmatched closing parenthesis according to the operation rules.
                # Such characters remain in the final string, so we append it to the `stack`.
                stack.append(char)
        else:
            # If the character is a lowercase English letter:
            # These characters are never removed by the operation.
            # Keep it in the result by appending it to the `stack`.
            stack.append(char)

    # After iterating through the entire input string S, the `stack` contains the sequence 
    # of characters that form the final processed string.
    # Join the characters in the stack list to form the final output string.
    # Using "".join() is an efficient way to construct a string from a list of characters.
    print("".join(stack))

# Execute the solve function to run the program logic.
solve()