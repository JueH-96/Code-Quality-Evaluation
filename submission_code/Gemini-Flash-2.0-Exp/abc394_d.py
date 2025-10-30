def solve():
    s = input()
    stack = []
    mapping = {')': '(', ']': '[', '>': '<'}
    
    for char in s:
        if char in ['(', '[', '<']:
            stack.append(char)
        elif char in [')', ']', '>']:
            if not stack:
                print("No")
                return
            top = stack.pop()
            if mapping[char] != top:
                print("No")
                return
        else:
            print("No")
            return
    
    if not stack:
        print("Yes")
    else:
        print("No")

solve()