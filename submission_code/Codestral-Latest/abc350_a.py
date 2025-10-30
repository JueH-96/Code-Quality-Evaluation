# YOUR CODE HERE
S = input().strip()

# List of valid abbreviations
valid_abbreviations = [f"ABC{i:03d}" for i in range(1, 315)] + [f"ABC{i:03d}" for i in range(317, 350)]

# Check if the input string is in the list of valid abbreviations
if S in valid_abbreviations:
    print("Yes")
else:
    print("No")