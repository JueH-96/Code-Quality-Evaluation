import sys

# Read the inputs from stdin
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

stack = []

for char in S:
    if 'a' <= char <= 'z' or char == '(':
        stack.append(char)
    elif char == ')':
        content_between_parens = []
        found_open = False
        
        # Pop elements until we find '(' or stack is empty
        while stack:
            top = stack[-1] # Peek
            if top == '(':
                found_open = True
                break
            else:
                # Pop and collect elements between potential '(' and ')'
                content_between_parens.append(stack.pop()) 

        if found_open:
            # Found a potential matching '(', check the collected content
            is_pure_letters = True
            for elem in content_between_parens:
                # Check if the element is a single lowercase letter
                if not ('a' <= elem <= 'z'):
                    is_pure_letters = False
                    break

            if is_pure_letters:
                # Found (letters). This pattern is deleted.
                # Pop the matching '(' from the stack.
                stack.pop() 
                # content_between_parens is discarded automatically
            else:
                # Found '(', but content was not pure letters (contained '(' or ')').
                # This ')' does not form a deletable pattern with this '('.
                # Restore the popped elements (content_between_parens) back to the stack.
                content_between_parens.reverse()
                stack.extend(content_between_parens)
                # The '(' is still on the stack, which is correct as it's not deleted.
                # Append the current ')' as it's not deleted either.
                stack.append(')')
        else:
            # Stack became empty, no matching '(' found for this ')'.
            # This ')' is an unmatched closing parenthesis according to the rule.
            # Restore any elements that were popped before stack became empty.
            content_between_parens.reverse()
            stack.extend(content_between_parens)
            # Append the current ')' as it's not deleted.
            stack.append(')')

# Join the remaining elements in the stack to form the final string
sys.stdout.write("".join(stack) + "
")