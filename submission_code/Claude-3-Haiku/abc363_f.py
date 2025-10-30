import re

def soln(N):
    # Convert N to a string
    N_str = str(N)
    
    # Generate all possible palindromes of length between 1 and 1000
    for length in range(1, 1001):
        for i in range(10**(length//2)):
            # Construct the palindrome
            if length % 2 == 0:
                s = str(i) + str(i)[::-1]
            else:
                s = str(i) + str(i)[:-1][::-1]
            
            # Check if the palindrome satisfies the conditions
            if s[0] in '123456789' and eval(s) == N and re.match(r'^[1-9*]+$', s):
                return s
    
    # If no palindrome satisfies the conditions, return -1
    return '-1'

# Read input from stdin
N = int(input())

# Print the solution
print(soln(N))