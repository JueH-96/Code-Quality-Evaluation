def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # The first line is N, which we don’t really need.
    # The second line is the string S.
    S = data[1].rstrip("
") if len(data) >= 2 else data[0].rstrip("
")
    
    # We will simulate the deletions with a stack.
    # The idea is that whenever we see a closing parenthesis ')',
    # we check if the characters at the end of our current built string (stored in stack)
    # form a pattern: an open parenthesis '(' immediately followed (to its right) by 
    # a contiguous block of letters (possibly empty). If so, the substring “(…)",
    # which is removable, gets deleted.
    #
    # More precisely, when we process a ')':
    #   - Let j = len(stack). Then while j > 0 and stack[j-1] is a lowercase letter,
    #     move j one step back.
    #   - If j > 0 and stack[j-1] is '(' then the pattern is detected.
    #     We remove all stack elements from index (j-1) onward.
    #   - Otherwise (i.e. no such open '(' was found immediately preceding the contiguous letters),
    #     we simply push the ')' onto the stack.
    #
    # This greedy strategy is equivalent to performing the allowed deletion operation repeatedly.
    
    stack = []
    for ch in S:
        if ch != ')':
            # For letters and '(' just push onto the stack.
            stack.append(ch)
        else:
            # When ch == ')', check if we can delete a valid substring.
            j = len(stack)
            # Move back over contiguous letters (a valid removable block may even be empty)
            while j > 0 and stack[j - 1].islower():
                j -= 1
            # If we found an open parenthesis right before that block then the pattern is valid.
            if j > 0 and stack[j - 1] == '(':
                # Remove the removal candidate (the open parenthesis and the letters)
                del stack[j - 1:]
            else:
                # Otherwise, no valid removal; so keep this ')'.
                stack.append(')')
    
    # The final string after maximal deletion
    sys.stdout.write("".join(stack))

if __name__ == '__main__':
    main()