# YOUR CODE HERE
n = int(input())
s = input()

while True:
    found = False
    i = 0
    while i < len(s):
        if s[i] == '(':
            # Find the first ')' after this '(' that doesn't have another '(' in between
            j = i + 1
            while j < len(s) and s[j] != ')' and s[j] != '(':
                j += 1
            
            if j < len(s) and s[j] == ')':
                # Remove substring from i to j inclusive
                s = s[:i] + s[j+1:]
                found = True
                break
        i += 1
    
    if not found:
        break

print(s)