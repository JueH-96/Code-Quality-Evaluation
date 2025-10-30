# YOUR CODE HERE
S = input().strip()

# Extract the numeric part (last 3 characters)
contest_num = int(S[3:])

# Check if it's a valid contest number
if (1 <= contest_num <= 315) or (317 <= contest_num <= 349):
    print("Yes")
else:
    print("No")