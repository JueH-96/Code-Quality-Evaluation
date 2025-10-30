def main():
    S = input().strip()
    # A stack to keep track of opening brackets
    stack = []
    # Mapping of closing to opening brackets
    match = {')': '(', ']': '[', '>': '<'}
    
    for c in S:
        if c in match.values():  # if c is one of '(', '[', '<'
            stack.append(c)
        else:  # c is one of ')', ']', '>'
            if not stack or stack[-1] != match[c]:
                print("No")
                return
            stack.pop()
    
    # If stack is empty at the end, it's valid
    print("Yes" if not stack else "No")

if __name__ == "__main__":
    main()