# Read the number of test cases
t = int(input())

for _ in range(t):
    # Read the input string
    s = input()
    
    # Check if the string is already "abc"
    if s == "abc":
        print("YES")
        continue
    
    # Check if the string can be converted to "abc" by swapping two characters
    if "".join(sorted(s)) == "abc":
        print("YES")
    else:
        print("NO")