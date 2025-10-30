# YOUR CODE HERE
def remove_parentheses(s):
    stack = []
    to_remove = set()
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            if all(c.islower() or c in to_remove for c in s[start+1:i]):
                to_remove.update(range(start, i+1))
    
    return ''.join(c for i, c in enumerate(s) if i not in to_remove)

N = int(input())
S = input().strip()

while True:
    new_S = remove_parentheses(S)
    if new_S == S:
        break
    S = new_S

print(S)