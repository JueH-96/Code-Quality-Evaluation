import sys

def is_11_22_string(s):
    n = len(s)
    if n % 2 == 0:
        return "No"
    
    mid = n // 2
    
    # Check if the first half is all 1s
    for i in range(mid - 1):
        if s[i] != '1':
            return "No"
    
    # Check if the middle character is '/'
    if s[mid] != '/':
        return "No"
    
    # Check if the second half is all 2s
    for i in range(mid + 1, n):
        if s[i] != '2':
            return "No"
    
    return "Yes"

# Read input from stdin
n = int(sys.stdin.readline().strip())
s = sys.stdin.readline().strip()

# Solve the problem
result = is_11_22_string(s)

# Write output to stdout
print(result)