# Read the input string
S = input().strip()

# Check if the sorted version of S equals "ABC"
if ''.join(sorted(S)) == 'ABC':
    print("Yes")
else:
    print("No")