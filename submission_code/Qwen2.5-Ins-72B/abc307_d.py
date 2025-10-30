# YOUR CODE HERE
def remove_parentheses(S):
    stack = []
    result = []
    for char in S:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                result.append(char)
        else:
            if not stack:
                result.append(char)
    return ''.join(result)

N = int(input())
S = input()
print(remove_parentheses(S))