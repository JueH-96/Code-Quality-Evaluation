# Read the input string
S = input().strip()

# Check if the sorted version of S matches the sorted target "ABC"
if sorted(S) == sorted("ABC"):
    print("Yes")
else:
    print("No")