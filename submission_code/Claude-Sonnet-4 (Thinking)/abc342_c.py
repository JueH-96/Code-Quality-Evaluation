# Read inputs
N = int(input())
S = input()
Q = int(input())

# Initialize mapping
mapping = {}
for c in 'abcdefghijklmnopqrstuvwxyz':
    mapping[c] = c

# Apply operations to the mapping
for _ in range(Q):
    c, d = input().split()
    # For all characters that currently map to c, change them to map to d
    for key in mapping:
        if mapping[key] == c:
            mapping[key] = d

# Apply the final mapping to the string
result = ''.join(mapping[char] for char in S)
print(result)