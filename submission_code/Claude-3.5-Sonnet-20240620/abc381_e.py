# YOUR CODE HERE
def max_11_22_subsequence(s):
    ones = twos = slashes = 0
    for char in s:
        if char == '1':
            ones += 1
        elif char == '2':
            twos += 1
        else:  # char == '/'
            slashes += 1
    
    if slashes == 0:
        return 0
    
    max_length = min(ones, twos) * 2 + 1
    return min(max_length, len(s))

# Read input
N, Q = map(int, input().split())
S = input().strip()

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    substring = S[L-1:R]
    result = max_11_22_subsequence(substring)
    print(result)