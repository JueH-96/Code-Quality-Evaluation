# Read the number of characters N and number of queries Q
N, Q = map(int, input().split())

# Read the string S
S = input().strip()

# Process each query
for _ in range(Q):
    l, r = map(int, input().split())
    
    # Convert to 0-based indexing
    l -= 1
    r -= 1
    
    # Count consecutive identical characters in the substring
    count = 0
    for i in range(l, r):
        if S[i] == S[i+1]:
            count += 1
    
    # Output the result
    print(count)