def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    s = data[1].rstrip("
")
    
    # We simulate the removal process using a stack.
    # We push characters onto the stack. When we see a ')', we try to remove a removable substring.
    # Our removal pattern is: it must start with '(' and end with ')', and in between contain only letters.
    # On encountering a ')', we pop any letters following the most recent '(' marker.
    # If no matching '(' is found, then ')' remains in the string.
    stack = []
    for ch in s:
        if ch == ')':
            temp = []
            found = False
            # Try to pop letters until we find a '(' marker.
            while stack:
                top = stack.pop()
                if top == '(':
                    found = True
                    break
                else:
                    # Letters popped, temporarily stored.
                    temp.append(top)
            if not found:
                # No matching '(' found; push back the popped letters in their original order,
                # then push the current ')'.
                for c in reversed(temp):
                    stack.append(c)
                stack.append(')')
            # If found, we have removed the substring (letters) including both the '(' and ')'
            # by not pushing anything back.
        else:
            # For letters and '(' simply push onto the stack.
            stack.append(ch)
    
    # The remaining stack forms the final string.
    sys.stdout.write("".join(stack))


if __name__ == '__main__':
    main()