def remove_parentheses(N, S):
    stack = []
    result = []

    for char in S:
        if char == '(':
            stack.append(len(result))  # Record the position in result where '(' is added
            result.append(char)
        elif char == ')':
            if stack:  # There is a matching '('
                start = stack.pop()  # Get the position of the matching '('
                # Remove the last added '(' from result
                result[start] = ''
                result.append('')  # Add an empty string for the ')'
        else:
            result.append(char)

    # Join the result and filter out empty strings
    final_result = ''.join(filter(lambda x: x != '', result))
    return final_result

import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]

print(remove_parentheses(N, S))