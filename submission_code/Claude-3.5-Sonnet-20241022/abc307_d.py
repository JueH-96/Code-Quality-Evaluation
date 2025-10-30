def find_removable_substring(s):
    n = len(s)
    for i in range(n-1):
        if s[i] == '(':
            # Find matching closing bracket
            count = 1
            j = i + 1
            while j < n and count > 0:
                if s[j] == '(':
                    count += 1
                elif s[j] == ')':
                    count -= 1
                j += 1
            
            # Check if we found a valid substring
            if count == 0 and j-1 > i:
                # Check if substring contains no brackets except ends
                inner = s[i+1:j-1]
                if '(' not in inner and ')' not in inner:
                    return (i, j)
    return None

N = int(input())
S = input()

while True:
    pos = find_removable_substring(S)
    if pos is None:
        break
    start, end = pos
    S = S[:start] + S[end:]

print(S)