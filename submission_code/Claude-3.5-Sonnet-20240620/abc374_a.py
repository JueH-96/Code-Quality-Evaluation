# YOUR CODE HERE
def check_san_suffix(s):
    return "Yes" if s.endswith("san") else "No"

# Read input from stdin
s = input().strip()

# Check if the string ends with "san" and print the result
print(check_san_suffix(s))