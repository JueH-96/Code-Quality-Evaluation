n = int(input())
s = list(input())
q = int(input())

for _ in range(q):
    parts = input().split()
    t = int(parts[0])
    if t == 1:
        x = int(parts[1]) - 1  # Convert to 0-based index
        c = parts[2]
        s[x] = c
    elif t == 2:
        # Convert all uppercase letters to lowercase
        s = [c.lower() if c.isupper() else c for c in s]
    elif t == 3:
        # Convert all lowercase letters to uppercase
        s = [c.upper() if c.islower() else c for c in s]

print(''.join(s))