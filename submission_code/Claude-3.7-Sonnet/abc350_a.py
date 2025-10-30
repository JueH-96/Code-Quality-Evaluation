# YOUR CODE HERE
s = input().strip()

# Extract contest number
contest_number = int(s[3:])

# Check if it's a valid and concluded contest
if 1 <= contest_number <= 349 and contest_number != 316:
    print("Yes")
else:
    print("No")