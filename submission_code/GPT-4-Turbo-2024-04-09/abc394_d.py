def is_colorful_bracket_sequence(s):
    stack = []
    matching_bracket = {')': '(', ']': '[', '>': '<'}
    
    for char in s:
        if char in matching_bracket.values():
            stack.append(char)
        elif char in matching_bracket:
            if not stack or stack[-1] != matching_bracket[char]:
                return "No"
            stack.pop()
    
    return "Yes" if not stack else "No"

import sys
input = sys.stdin.read
s = input().strip()
print(is_colorful_bracket_sequence(s))