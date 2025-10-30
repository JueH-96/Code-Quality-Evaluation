def is_1122_string(s):
    # Check if string s can form a valid 11/22 string
    ones = 0  # count of 1s
    twos = 0  # count of 2s
    slashes = 0  # count of /
    
    for c in s:
        if c == '1':
            ones += 1
        elif c == '2':
            twos += 1
        else:  # c == '/'
            slashes += 1
    
    # For a valid 11/22 string:
    # - We need exactly one slash
    # - Number of 1s and 2s should be equal
    # - Total length should be odd
    if slashes != 1:
        return 0
    
    # Return maximum possible length of 11/22 string
    # that can be formed with these characters
    min_side = min(ones, twos)
    if min_side == 0:
        return 1 if slashes == 1 else 0  # only '/' is valid
    return 2 * min_side + 1  # ones + slash + twos

def solve_query(s, l, r):
    # Get substring (0-based indexing)
    substr = s[l-1:r]
    return is_1122_string(substr)

# Read input
N, Q = map(int, input().split())
S = input().strip()

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    result = solve_query(S, L, R)
    print(result)