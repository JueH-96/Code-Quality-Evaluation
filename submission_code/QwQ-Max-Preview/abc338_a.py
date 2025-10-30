# Read the input string
S = input().strip()

# Check the conditions
first_char_upper = S[0].isupper()
rest_lower = all(c.islower() for c in S[1:])

if first_char_upper and rest_lower:
    print("Yes")
else:
    print("No")