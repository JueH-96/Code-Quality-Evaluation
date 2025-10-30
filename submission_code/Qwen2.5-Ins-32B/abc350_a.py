S = input().strip()

# Define the valid range of contests
valid_contests = set(range(1, 316)) | set(range(317, 350))

# Extract the number part from the string
contest_number = int(S[3:])

# Check if the contest number is valid
if contest_number in valid_contests:
    print("Yes")
else:
    print("No")