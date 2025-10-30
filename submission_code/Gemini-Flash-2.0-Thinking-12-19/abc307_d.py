def solve():
    n = int(input())
    s = list(input())
    
    while True:
        found_and_removed = False
        i = 0
        while i < len(s):
            if s[i] == '(':
                j = -1
                for k in range(i + 1, len(s)):
                    if s[k] == ')':
                        j = k
                        break
                if j != -1:
                    substring_between = s[i+1:j]
                    is_lowercase_only = True
                    for char in substring_between:
                        if not 'a' <= char <= 'z':
                            is_lowercase_only = False
                            break
                    if is_lowercase_only:
                        s = s[:i] + s[j+1:]
                        found_and_removed = True
                        break
            i += 1
        if not found_and_removed:
            break
            
    print("".join(s))

if __name__ == '__main__':
    solve()