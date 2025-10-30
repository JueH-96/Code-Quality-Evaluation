n = int(input())
s = input()

def find_removable():
    stack = []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            if all(c.islower() for c in s[start+1:i]):
                return start, i
    return None

while True:
    pos = find_removable()
    if pos is None:
        break
    start, end = pos
    s = s[:start] + s[end+1:]

print(s)