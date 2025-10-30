s = input().strip()

i = 0
n = len(s)

# Skip all consecutive A's from the beginning
while i < n and s[i] == 'A':
    i += 1

# Skip all consecutive B's after A's
while i < n and s[i] == 'B':
    i += 1

# Skip all consecutive C's after B's
while i < n and s[i] == 'C':
    i += 1

# Check if we processed the entire string
if i == n:
    print("Yes")
else:
    print("No")