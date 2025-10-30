n = int(input())
s = input()
q = int(input())

# Initialize mapping - each character maps to itself
mapping = [chr(i + ord('a')) for i in range(26)]

# Process each operation
for _ in range(q):
    c, d = input().split()
    # Update mapping: all characters that currently map to c should now map to d
    for i in range(26):
        if mapping[i] == c:
            mapping[i] = d

# Apply final mapping to string
result = []
for char in s:
    idx = ord(char) - ord('a')
    result.append(mapping[idx])

print(''.join(result))