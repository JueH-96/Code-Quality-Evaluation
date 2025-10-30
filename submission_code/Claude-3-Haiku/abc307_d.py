def remove_nested_parentheses(s):
    stack = []
    result = []
    for c in s:
        if c == '(':
            stack.append(c)
        elif c == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                result.append(c)
        else:
            result.append(c)
    return ''.join(result)

n = int(input())
s = input()
print(remove_nested_parentheses(s))