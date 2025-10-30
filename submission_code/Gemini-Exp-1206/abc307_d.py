def solve():
    n = int(input())
    s = input()
    
    while True:
        found = False
        new_s = ""
        i = 0
        while i < len(s):
            if s[i] == '(':
                j = i + 1
                while j < len(s) and s[j] != '(' and s[j] != ')':
                    j += 1
                if j < len(s) and s[j] == ')':
                    found = True
                    i = j + 1
                else:
                    new_s += s[i]
                    i += 1
            else:
                new_s += s[i]
                i += 1
        
        if not found:
            break
        s = new_s
    
    print(s)

solve()