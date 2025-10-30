import sys

def solve():
    """
    Reads a string from standard input and determines if it is a
    colorful bracket sequence using a stack-based approach.
    """
    # Read the input string from stdin. Using sys.stdin.readline is generally
    # faster for large inputs, and .strip() removes any trailing newline.
    S = sys.stdin.readline().strip()

    # Use a list as a stack to keep track of opening brackets.
    stack = []

    # A dictionary to map opening brackets to their corresponding closing ones.
    # This makes checking for a valid pair easy and efficient.
    pairs = {
        '(': ')',
        '[': ']',
        '<': '>'
    }

    # Iterate through each character of the input string S.
    for char in S:
        # If the character is an opening bracket (a key in our `pairs` map),
        # push it onto the stack.
        if char in pairs:
            stack.append(char)
        # Otherwise, the character must be a closing bracket.
        else:
            # If the stack is empty, this closing bracket has no matching
            # opening bracket, so the sequence is invalid.
            if not stack:
                print("No")
                # Exit the program since we have our answer.
                return
            
            # Pop the most recent opening bracket from the stack.
            last_open_bracket = stack.pop()
            
            # Check if the current closing bracket matches the popped one.
            # If they don't form a valid pair (e.g., opening '[' and closing ')'),
            # the sequence is invalid.
            if pairs[last_open_bracket] != char:
                print("No")
                # Exit the program since we have our answer.
                return

    # After checking all characters, if the stack is empty, every opening bracket
    # was correctly matched. The sequence is valid.
    if not stack:
        print("Yes")
    # If the stack is not empty, there are unclosed opening brackets,
    # so the sequence is invalid.
    else:
        print("No")

solve()