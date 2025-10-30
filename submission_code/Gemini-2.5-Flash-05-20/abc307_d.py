import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    stack = []

    for char_s in S:
        if char_s == '(':
            stack.append(char_s)
        elif char_s == ')':
            is_valid_segment_content = True
            temp_segment_chars = []
            open_paren_found_for_deletion = False

            # Pop elements from the stack until an opening parenthesis '(' is found or the stack is empty.
            while stack:
                char_from_stack = stack.pop()
                if char_from_stack == '(':
                    open_paren_found_for_deletion = True
                    break # Found the matching opening parenthesis
                else: # char_from_stack is a lowercase letter or another ')'
                    # If the content of the segment (X in (X)) contains another ')'
                    # then this segment is not "clean" and cannot be deleted.
                    if char_from_stack == ')':
                        is_valid_segment_content = False
                    temp_segment_chars.append(char_from_stack)
            
            # Decide whether to delete the (X) segment or restore elements
            if open_paren_found_for_deletion and is_valid_segment_content:
                # The (X) segment is valid for deletion.
                # Since the '(' and 'X's characters have already been popped, we do nothing.
                pass 
            else:
                # The segment is not valid for deletion (either no matching '(' or content not clean).
                # Restore the stack to its state before processing this ')' and the characters that were part of the potential segment.
                
                # If an opening parenthesis was found but the segment was not clean, push it back.
                if open_paren_found_for_deletion:
                    stack.append('(')
                
                # Push back the characters that were temporarily popped from the segment, in reverse order.
                for c in reversed(temp_segment_chars):
                    stack.append(c)
                
                # Finally, push the current ')' character onto the stack, as it's not deleted.
                stack.append(char_s) 
        else: # char_s is a lowercase English letter
            stack.append(char_s)
            
    # Join the remaining characters in the stack to form the final string.
    sys.stdout.write("".join(stack) + "
")

solve()