def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Use a stack to check if S is a colorful bracket sequence.
    # A colorful bracket sequence behaves like a correctly nested bracket sequence.
    # We remove pairs whenever we encounter one of "()", "[]", or "<>".
    
    stack = []
    # Dictionary for matching closing to opening brackets.
    matching = {')': '(', ']': '[', '>': '<'}
    
    # Process each character in the string.
    for ch in S:
        # For an opening bracket, push it onto the stack.
        if ch in "([<":
            stack.append(ch)
        else:
            # If there is a closing bracket, either the stack must not be empty 
            # and the most recent opening bracket at the top of the stack must match.
            if not stack or stack[-1] != matching[ch]:
                print("No")
                return
            stack.pop()
    
    # If the stack is empty, then every opening bracket was properly matched.
    print("Yes" if not stack else "No")

if __name__ == '__main__':
    main()