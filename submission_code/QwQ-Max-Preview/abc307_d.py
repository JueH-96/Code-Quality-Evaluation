n = int(input())
s = input().strip()

while True:
    # Compute the prefix sum array of parentheses count
    current_length = len(s)
    prefix = [0] * (current_length + 1)
    for i in range(current_length):
        prefix[i+1] = prefix[i] + (1 if s[i] in '()' else 0)
    
    stack = []
    valid_pairs = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                # Check if there are any parentheses between j+1 and i-1
                if prefix[i] - prefix[j+1] == 0:
                    valid_pairs.append( (j, i) )
    
    if not valid_pairs:
        break
    
    # Remove the valid pairs from the string; process in reverse order to avoid index issues
    valid_pairs.sort(reverse=True, key=lambda x: x[0])
    new_s = list(s)
    for start, end in valid_pairs:
        new_s[start:end+1] = []
    s = ''.join(new_s)

print(s)