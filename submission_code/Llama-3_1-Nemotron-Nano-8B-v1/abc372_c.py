n, q = map(int, input().split())
s = list(input().strip())
count = 0

# Initialize the count of ABC triplets
for i in range(n - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        count += 1

for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-based index

    # Check and subtract affected triplets before the change
    # Triplet where x is the first character (start = x)
    start = x
    if start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count -= 1
    # Triplet where x is the second character (start = x-1)
    start = x - 1
    if start >= 0 and start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count -= 1
    # Triplet where x is the third character (start = x-2)
    start = x - 2
    if start >= 0 and start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count -= 1

    # Update the character
    s[x] = c

    # Check and add affected triplets after the change
    # Triplet where x is the first character
    start = x
    if start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count += 1
    # Triplet where x is the second character
    start = x - 1
    if start >= 0 and start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count += 1
    # Triplet where x is the third character
    start = x - 2
    if start >= 0 and start + 2 < n:
        if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
            count += 1

    print(count)