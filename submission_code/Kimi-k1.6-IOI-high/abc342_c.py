n = int(input())
s = input().strip()
q = int(input())

# Initialize the map where each character maps to itself
char_map = list(range(26))  # 0-'a', 1-'b', ..., 25-'z'

for _ in range(q):
    c, d = input().split()
    c_idx = ord(c) - ord('a')
    d_idx = ord(d) - ord('a')
    # Update all characters that currently map to c_idx to map to d_idx
    for x in range(26):
        if char_map[x] == c_idx:
            char_map[x] = d_idx

# Build the result string using the final mapping
result = []
for ch in s:
    original_idx = ord(ch) - ord('a')
    new_idx = char_map[original_idx]
    result.append(chr(new_idx + ord('a')))

print(''.join(result))