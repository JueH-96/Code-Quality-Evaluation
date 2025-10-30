def is_good_string(s):
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            return False
    return True

# Read the input
N, Q = map(int, input().split())
S = list(input().strip())

# Process the queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1 query: Flip characters
        L, R = query[1], query[2]
        for i in range(L-1, R):
            S[i] = '1' if S[i] == '0' else '0'
    else:
        # Type 2 query: Check if the substring is a good string
        L, R = query[1], query[2]
        if is_good_string(S[L-1:R]):
            print("Yes")
        else:
            print("No")