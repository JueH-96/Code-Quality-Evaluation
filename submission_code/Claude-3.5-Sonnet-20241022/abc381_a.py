# Read input
N = int(input())
S = input()

def is_11_22_string(s, n):
    # Length must be odd
    if n % 2 == 0:
        return False
    
    # Calculate middle position
    mid = n // 2
    
    # Check if middle character is '/'
    if s[mid] != '/':
        return False
    
    # Check if all characters before middle are '1'
    for i in range(mid):
        if s[i] != '1':
            return False
    
    # Check if all characters after middle are '2'
    for i in range(mid + 1, n):
        if s[i] != '2':
            return False
    
    return True

# Check if S is an 11/22 string and print result
if is_11_22_string(S, N):
    print("Yes")
else:
    print("No")