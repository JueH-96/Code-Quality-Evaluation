def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    S = data[1].strip() if len(data) > 1 else ""
    
    # We will simulate the removal operations using a stack.
    #
    # Whenever we see a character:
    #   - If it's a letter or an opening parenthesis "(", simply push it onto the stack.
    #   - If it's a closing parenthesis ")", we try to see if it forms a valid removable segment.
    #
    # A removable segment is defined as a contiguous substring that starts with "(" and ends with ")"
    # and contains no other parentheses in-between. In our simulation, when we encounter a ")",
    # we pop from the stack until we hit the matching "(".
    # If there is no interference by extra parentheses, we remove the entire segment.
    #
    # This simulates removal of the innermost valid substring, and since the final outcome is 
    # uniquely determined, it is sufficient.
    
    stack = []
    for ch in S:
        if ch == ')':
            temp = []
            valid = True
            # Remove characters until we hit an opening '('
            while stack and stack[-1] != '(':
                top = stack.pop()
                # If by any chance a parenthesis appears here, the segment would be invalid,
                # but if our simulation is correct, we should see only letters here.
                if top == '(' or top == ')':
                    valid = False
                    break
                temp.append(top)
            # Check if we indeed found an opening "(" and the substring was valid.
            if stack and stack[-1] == '(' and valid:
                stack.pop()  # Remove the opening parenthesis.
                # We do not push the content back to stack; the entire segment is removed.
            else:
                # If not valid (or no matching "(" found), restore what we popped (if any)
                for x in reversed(temp):
                    stack.append(x)
                # And treat the current ")" as ordinary, so push it.
                stack.append(ch)
        else:
            # For letters and '(' add them directly
            stack.append(ch)
    
    result = "".join(stack)
    sys.stdout.write(result)

if __name__ == '__main__':
    main()