def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    stack = []
    for c in S:
        if c != ')':
            # Just push if it's a lowercase letter or '('
            stack.append(c)
        else:
            # We attempt to remove a substring of the form (letters)
            temp = []
            # Pop all trailing letters
            while stack and stack[-1].isalpha():
                temp.append(stack.pop())
            # Now see if there's a matching '(' on top
            if stack and stack[-1] == '(':
                # We found a valid substring "(...)" with only letters in between
                # Remove the '(' and do not push this ')', effectively deleting the substring
                stack.pop()
            else:
                # Not a valid substring to remove; restore all popped chars + push ')'
                while temp:
                    stack.append(temp.pop())
                stack.append(c)
    
    print(''.join(stack))

if __name__ == "__main__":
    main()