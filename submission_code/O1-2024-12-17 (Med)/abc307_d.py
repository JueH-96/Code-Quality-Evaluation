def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    
    # We will use a stack of "tokens". Each token is either:
    #   1) A string of lowercase letters (no parentheses)
    #   2) '('
    #
    # Whenever we see a ')', we try to pop until we find '('. 
    #   - If among the popped tokens there is any '(' token (besides the final one we match)
    #     or if we cannot find an '(' at all, then we cannot remove that substring
    #     (i.e. it contains nested parentheses or is unmatched), so we revert the popped items
    #     plus we push the literal ')' onto the stack.
    #   - Otherwise, we found '(' and only letter-tokens in between, so we remove them all
    #     (i.e. do not push them back, effectively deleting "(...)" ).
    #
    # In the end, we print the content of the stack as-is.
    #
    # This procedure yields the unique final string after all possible removals.

    stack = []
    
    def push_string(stk, txt):
        """Push a string of letters onto the stack, merging with top if that is also a string."""
        if stk and isinstance(stk[-1], str):
            stk[-1] += txt
        else:
            stk.append(txt)

    for c in S:
        if c.isalpha():
            # c is a letter
            push_string(stack, c)
        elif c == '(':
            stack.append(c)
        else:  # c == ')'
            buffer = []
            blocked = False
            found_open = False
            
            # Pop until we find '(' or stack is empty
            while stack:
                top = stack.pop()
                if top == '(':
                    # we found a matching '('
                    found_open = True
                    break
                else:
                    # If it's a string of letters, that's fine; put it in buffer
                    if isinstance(top, str):
                        buffer.append(top)
                    else:
                        # It's another '(' or some unexpected token (in practice just '(' possible),
                        # so that means there's nesting
                        blocked = True
                        buffer.append(top)
                        # But we still need to continue popping until we see the '('
                        # so do NOT break here. We'll keep popping to find '(',
                        # but we already know it's blocked.
            
            if not found_open:
                # That means we never found '(' => blocked
                blocked = True
            
            if blocked:
                # revert everything including '(' if we popped it (and it wasn't our matching one),
                # but note we only pop until we see '(' or empty. We did see '(' if found_open = True, 
                # but that was presumably the matching one; we must push it back as well if blocked.
                # If we didn't find_open, there's no '(' to push back.
                
                # If found_open, push '(' back
                if found_open:
                    stack.append('(')
                
                # revert buffer in reverse order so they appear in original order on stack
                while buffer:
                    stack.append(buffer.pop())
                
                # finally push the ')'
                stack.append(')')
            else:
                # found '(' and not blocked => remove the substring
                # do nothing => effectively deleting the "( ... )"
                pass
    
    # Now convert the stack to the final string.
    # stack may contain string tokens and '(' tokens and possibly ')' tokens.
    # Just join them together.
    print("".join(stack))

# Call main() at the end
if __name__ == "__main__":
    main()