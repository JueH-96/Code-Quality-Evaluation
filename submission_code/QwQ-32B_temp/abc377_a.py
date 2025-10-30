# Read the input string
s = input().strip()

# Check if the sorted characters match 'A', 'B', 'C'
if sorted(s) == ['A', 'B', 'C']:
    print("Yes")
else:
    print("No")