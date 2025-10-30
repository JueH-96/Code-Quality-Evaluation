# Read input
n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

# Function to check if a string is a palindrome
def is_palindrome(s):
    return s == s[::-1]

# Check all pairs (i, j) where i != j
found = False
for i in range(n):
    for j in range(n):
        if i != j:
            # Concatenate strings[i] and strings[j]
            concatenated = strings[i] + strings[j]
            if is_palindrome(concatenated):
                found = True
                break
    if found:
        break

# Output result
if found:
    print("Yes")
else:
    print("No")