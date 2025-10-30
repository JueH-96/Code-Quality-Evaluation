# Read the input values
N = int(input().strip())
S = list(input().strip())  # Convert string to list for easier manipulation
Q = int(input().strip())

# Process each operation
for _ in range(Q):
    t, x, c = input().strip().split()
    t = int(t)
    if t == 1:
        x = int(x) - 1  # Convert to zero-based index
        S[x] = c
    elif t == 2:
        S = [char.lower() if char.isupper() else char for char in S]
    elif t == 3:
        S = [char.upper() if char.islower() else char for char in S]

# Convert the list back to a string and print the result
print(''.join(S))