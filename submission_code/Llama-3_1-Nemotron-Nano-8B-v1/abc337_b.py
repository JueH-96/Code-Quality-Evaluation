# Read the input string
S = input().strip()

i = 0
# Determine the end of the A part
while i < len(S) and S[i] == 'A':
    i += 1

j = i
# Determine the end of the B part
while j < len(S) and S[j] == 'B':
    j += 1

k = j
# Determine the end of the C part
while k < len(S) and S[k] == 'C':
    k += 1

# Check if all parts are processed correctly
print("Yes" if k == len(S) else "No")