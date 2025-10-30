def is_1122_string(s):
    n = len(s)
    if n % 2 == 0:  # length must be odd
        return False
    
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

# Read input
N = int(input())
S = input()

max_length = 1  # minimum length is 1 (just the '/' character)

# Try all possible substrings
for i in range(N):
    for length in range(1, N - i + 1, 2):  # only try odd lengths
        substr = S[i:i+length]
        if '/' in substr and is_1122_string(substr):
            max_length = max(max_length, length)

print(max_length)