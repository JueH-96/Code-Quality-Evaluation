# YOUR CODE HERE
n = int(input())
s = input()

# Convert string to list for easier manipulation
chars = list(s)
removed = [False] * n

while True:
    found = False
    
    # Look for patterns to remove
    i = 0
    while i < n:
        if not removed[i] and chars[i] == '(':
            # Found an opening parenthesis
            j = i + 1
            
            # Look for the matching closing parenthesis
            # ensuring no other parentheses in between
            valid = True
            while j < n and not removed[j]:
                if chars[j] == '(' or (chars[j] == ')' and j == i + 1):
                    valid = False
                    break
                elif chars[j] == ')':
                    # Found a valid pattern to remove
                    if valid:
                        # Mark all characters from i to j as removed
                        for k in range(i, j + 1):
                            removed[k] = True
                        found = True
                    break
                j += 1
            
            if found:
                break
        i += 1
    
    if not found:
        break

# Build the result string
result = []
for i in range(n):
    if not removed[i]:
        result.append(chars[i])

print(''.join(result))