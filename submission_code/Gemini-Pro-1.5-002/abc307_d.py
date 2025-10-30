# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    
    stack = []
    res = ""
    
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                res += char
        else:
            if not stack:
                res += char
    
    print(res)

solve()