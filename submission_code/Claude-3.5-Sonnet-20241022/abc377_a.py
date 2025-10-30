# Read input string
S = input().strip()

# Sort the characters of input string
sorted_S = ''.join(sorted(S))

# Check if sorted string equals "ABC"
if sorted_S == "ABC":
    print("Yes")
else:
    print("No")