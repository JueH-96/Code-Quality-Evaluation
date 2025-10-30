def solve():
    s = input()
    n = len(s)
    mod = 998244353

    q_count = s.count('?')
    ans = 0

    for i in range(2**q_count):
        temp = bin(i)[2:].zfill(q_count)
        curr = ""
        k=0
        for char in s:
            if char == '?':
                if temp[k] == '0':
                    curr += '('
                else:
                    curr += ')'
                k+=1
            else:
                curr += char
        
        
        
        def is_parenthesis_string(string):
            if not string:
                return True
            
            stack = []
            for char in string:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if not stack:
                        return False
                    stack.pop()
            
            return len(stack) == 0

        if is_parenthesis_string(curr):
            ans = (ans + 1) % mod

    print(ans)

solve()