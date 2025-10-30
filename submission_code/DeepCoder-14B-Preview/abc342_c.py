n = int(input())
s = input().strip()
q = int(input())

# Initialize transformation dictionary
trans = {chr(ord('a') + i): chr(ord('a') + i) for i in range(26)}

for _ in range(q):
    c, d = input().split()
    # Update all characters that currently map to c to now map to d
    for x in trans:
        if trans[x] == c:
            trans[x] = d

# Build the resulting string
result = []
for char in s:
    result.append(trans[char])

print(''.join(result))