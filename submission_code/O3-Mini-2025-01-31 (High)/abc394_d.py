def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    # The input S is provided in the first token.
    S = data.split()[0]
    
    # Define valid pairs: closing bracket maps to its corresponding opening.
    pairs = {')': '(', ']': '[', '>': '<'}
    
    stack = []
    
    for ch in S:
        # If ch is an opening bracket, push it onto the stack.
        if ch in "([<":
            stack.append(ch)
        else:
            # Otherwise, ch is a closing bracket.
            # Check if there is a corresponding opening bracket on top of the stack.
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            else:
                # Mismatch or no opening bracket available.
                print("No")
                return

    # If the stack is empty, then all brackets have been matched.
    print("Yes" if not stack else "No")
    
if __name__ == '__main__':
    main()