from collections import deque

def is_colorful_bracket_sequence(s):
    stack = deque()
    for c in s:
        if c in ['(', '[', '<']:
            stack.append(c)
        else:
            if not stack:
                return False
            top = stack.pop()
            if (top, c) not in [('(', ')'), ('[', ']'), ('<', '>')]:
                return False
    return len(stack) == 0

s = input()
print("Yes" if is_colorful_bracket_sequence(s) else "No")