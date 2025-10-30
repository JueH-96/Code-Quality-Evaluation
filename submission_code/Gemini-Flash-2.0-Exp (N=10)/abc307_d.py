def solve():
    n = int(input())
    s = input()
    
    stack = []
    result = []
    
    for char in s:
        if char == ')':
            if stack and stack[-1] == '(':
                stack.pop()
                while result and result[-1] != '(':
                    result.pop()
                if result and result[-1] == '(':
                    result.pop()
            else:
                result.append(char)
        else:
            result.append(char)
            if char == '(':
                stack.append(char)
                
    print("".join(result))

solve()