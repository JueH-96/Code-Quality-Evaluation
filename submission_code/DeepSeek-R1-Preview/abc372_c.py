n, q = map(int, input().split())
s = list(input().strip())
current_count = 0

# Calculate initial count of ABC
for i in range(len(s) - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        current_count += 1

for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # convert to 0-based index

    if s[x] == c:
        print(current_count)
        continue

    # Collect affected indices
    affected = []
    for di in (-2, -1, 0):
        i = x + di
        if 0 <= i <= len(s) - 3:
            affected.append(i)

    # Subtract old ABC counts
    for i in affected:
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            current_count -= 1

    # Update the character
    s[x] = c

    # Add new ABC counts
    for i in affected:
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            current_count += 1

    print(current_count)