# YOUR CODE HERE
S = input().strip()

# Find the end of the A section
i = 0
while i < len(S) and S[i] == 'A':
    i += 1

# Find the end of the B section
while i < len(S) and S[i] == 'B':
    i += 1

# Check if all remaining characters are C
is_extended_abc = True
while i < len(S):
    if S[i] != 'C':
        is_extended_abc = False
        break
    i += 1

if is_extended_abc:
    print("Yes")
else:
    print("No")