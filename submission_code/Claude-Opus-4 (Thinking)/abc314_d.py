# YOUR CODE HERE
n = int(input())
s = list(input())
q = int(input())

# Find the last type 2 or 3 operation
last_transform = -1
last_transform_type = 0
operations = []

for i in range(q):
    parts = input().split()
    t = int(parts[0])
    x = int(parts[1])
    c = parts[2]
    operations.append((t, x, c))
    
    if t == 2:
        last_transform = i
        last_transform_type = 2
    elif t == 3:
        last_transform = i
        last_transform_type = 3

# Apply operations
if last_transform == -1:
    # No transform operations
    for t, x, c in operations:
        if t == 1:
            s[x - 1] = c
else:
    # Apply operations before the last transform
    for i in range(last_transform):
        t, x, c = operations[i]
        if t == 1:
            s[x - 1] = c
    
    # Apply the last transform
    if last_transform_type == 2:
        # Convert all uppercase to lowercase
        for i in range(n):
            if s[i].isupper():
                s[i] = s[i].lower()
    else:  # last_transform_type == 3
        # Convert all lowercase to uppercase
        for i in range(n):
            if s[i].islower():
                s[i] = s[i].upper()
    
    # Apply operations after the last transform
    for i in range(last_transform + 1, q):
        t, x, c = operations[i]
        if t == 1:
            s[x - 1] = c

print(''.join(s))