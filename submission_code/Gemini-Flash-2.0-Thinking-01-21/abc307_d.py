def solve():
    n = int(input())
    s = list(input())
    
    while True:
        found_removal = False
        i = 0
        while i < len(s):
            if s[i] == '(':
                for j in range(i + 1, len(s)):
                    if s[j] == ')':
                        middle_part = s[i+1:j]
                        has_parentheses = False
                        for char in middle_part:
                            if char == '(' or char == ')':
                                has_parentheses = True
                                break
                        if not has_parentheses:
                            s = s[:i] + s[j+1:]
                            found_removal = True
                            break
                if found_removal:
                    break
            i += 1
        if not found_removal:
            break
            
    print("".join(s))

if __name__ == '__main__':
    solve()