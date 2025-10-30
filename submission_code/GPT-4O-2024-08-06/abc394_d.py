# YOUR CODE HERE
def is_colorful_bracket_sequence(S):
    stack = []
    matching_bracket = {')': '(', ']': '[', '>': '<'}
    
    for char in S:
        if char in '([{<':
            stack.append(char)
        elif char in ')]}>':
            if stack and stack[-1] == matching_bracket[char]:
                stack.pop()
            else:
                return "No"
    
    return "Yes" if not stack else "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(is_colorful_bracket_sequence(S))