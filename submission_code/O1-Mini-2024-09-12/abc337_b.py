S = input().strip()

i = 0
n = len(S)

# Traverse all 'A's
while i < n and S[i] == 'A':
    i += 1

# Traverse all 'B's
while i < n and S[i] == 'B':
    i += 1

# Traverse all 'C's
while i < n and S[i] == 'C':
    i += 1

# If we've reached the end, it's a valid Extended ABC string
if i == n:
    print("Yes")
else:
    print("No")