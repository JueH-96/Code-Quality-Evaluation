# Read the inputs
N, Q = map(int, input().split())
S = list(input())

# Process each query
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        # Flip the characters from L to R
        L, R = query[1] - 1, query[2]
        for i in range(L, R):
            S[i] = '0' if S[i] == '1' else '1'
    elif query[0] == 2:
        # Check if the substring is a good string
        L, R = query[1] - 1, query[2]
        is_good = all(S[i] != S[i + 1] for i in range(L, R - 1))
        print('Yes' if is_good else 'No')