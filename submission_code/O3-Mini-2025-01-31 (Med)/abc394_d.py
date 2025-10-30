def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Dictionary to map closing bracket to its corresponding opening bracket
    match = {')': '(', ']': '[', '>': '<'}
    
    stack = []
    
    for ch in S:
        if ch in "([<":
            # If it's an opening bracket, push it to the stack.
            stack.append(ch)
        else:
            # Otherwise, it must be a closing bracket.
            # If the stack is empty or the top doesn't match, it's not a valid sequence.
            if not stack or stack[-1] != match[ch]:
                print("No")
                return
            # Otherwise, pop the matching opening bracket.
            stack.pop()
    
    # After processing, if the stack is empty then the string is a valid "colorful" bracket sequence.
    print("Yes" if not stack else "No")

if __name__ == '__main__':
    main()