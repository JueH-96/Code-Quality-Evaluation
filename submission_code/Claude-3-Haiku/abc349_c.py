import sys

def is_airport_code(s, t):
    # Check if T can be derived from S by taking a subsequence of length 3
    for i in range(len(s) - 2):
        if s[i:i+3].upper() == t:
            return "Yes"
    
    # Check if T can be derived from S by taking a subsequence of length 2 and appending 'X'
    for i in range(len(s) - 1):
        for j in range(i+1, len(s)):
            if s[i:i+1].upper() + s[j:j+1].upper() == t[:-1] and t[-1] == 'X':
                return "Yes"
    
    return "No"

# Read input from stdin
s = input()
t = input()

# Print the result
print(is_airport_code(s, t))