def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]

    stack = []
    for c in S:
        if c != ')':
            # Just push any character that is not ')'
            stack.append(c)
        else:
            # We encountered a ')', try to match a '(' that encloses only letters
            content = []
            has_paren = False
            found_open_paren = False
            
            # Pop until we find a '(' or the stack is empty
            while stack and stack[-1] != '(':
                top = stack.pop()
                content.append(top)
                if top in ['(', ')']:
                    has_paren = True

            # Check if there's a matching '('
            if stack and stack[-1] == '(':
                stack.pop()  # Remove the '('
                found_open_paren = True

            if found_open_paren:
                # We matched '(', now see if the content contained any parentheses
                if has_paren:
                    # If it has parentheses, we can't remove this block, so push it back
                    stack.append('(')
                    # content is in reverse order, so re-push in correct order
                    while content:
                        stack.append(content.pop())
                    stack.append(')')
                # else: If no parentheses inside, do nothing (remove the substring)
            else:
                # No matching '(', push back what we popped plus this ')'
                while content:
                    stack.append(content.pop())
                stack.append(')')

    print("".join(stack))

# Do not forget to call main()
if __name__ == "__main__":
    main()