# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    
    while True:
        new_s = ""
        i = 0
        found = False
        
        while i < len(s):
            if s[i] == '(':
                j = i + 1
                valid = True
                
                while j < len(s) and s[j] != ')':
                    if s[j] == '(':
                        valid = False
                        break
                    j += 1
                
                if j < len(s) and s[j] == ')' and valid:
                    found = True
                    i = j + 1  # Skip this substring
                else:
                    new_s += s[i]
                    i += 1
            else:
                new_s += s[i]
                i += 1
        
        if not found:
            break
        s = new_s
    
    return s

print(solve())