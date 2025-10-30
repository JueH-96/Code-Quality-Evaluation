# Read the number of test cases
t = int(input())

# Iterate over each test case
for _ in range(t):
    # Read the string for the current test case
    s = input().strip()
    
    # Check if the string can be made "abc" with at most one swap
    if s == "abc" or s[1:] == "bc" or s[0] == "a":
        print("YES")
    else:
        print("NO")