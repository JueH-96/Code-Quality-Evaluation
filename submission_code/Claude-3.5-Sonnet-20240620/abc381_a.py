# YOUR CODE HERE
def is_11_22_string(s):
    n = len(s)
    
    # Check if length is odd
    if n % 2 == 0:
        return False
    
    mid = n // 2
    
    # Check if middle character is '/'
    if s[mid] != '/':
        return False
    
    # Check if first half is all '1's
    if not all(c == '1' for c in s[:mid]):
        return False
    
    # Check if second half is all '2's
    if not all(c == '2' for c in s[mid+1:]):
        return False
    
    return True

# Read input
N = int(input())
S = input().strip()

# Check if S is an 11/22 string
if is_11_22_string(S):
    print("Yes")
else:
    print("No")