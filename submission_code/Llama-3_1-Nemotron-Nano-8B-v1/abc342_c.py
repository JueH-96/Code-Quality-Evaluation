n = int(input())
s = input().strip()
q = int(input())

# Initialize current mapping for each lowercase letter to itself
current_map = {chr(ord('a') + i): chr(ord('a') + i) for i in range(26)}

for _ in range(q):
    c, d = input().split()
    # Update all characters that currently map to c to map to d
    for x in current_map:
        if current_map[x] == c:
            current_map[x] = d

# Build the result string using the current_map
result = [current_map[c] for c in s]
print(''.join(result))