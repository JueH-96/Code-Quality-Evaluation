# YOUR CODE HERE

# Read the inputs
N, Q = map(int, input().split())
S = input()

# Initialize a list to store the count of consecutive characters
count = [[0]*26 for _ in range(N+1)]

# Iterate over the string
for i in range(N):
    # Update the count of consecutive characters
    for j in range(26):
        if ord(S[i])-97 == j:
            count[i+1][j] = count[i][j] + 1
        else:
            count[i+1][j] = count[i][j]

# Process the queries
for _ in range(Q):
    l, r = map(int, input().split())
    ans = 0
    # Iterate over the alphabets
    for j in range(26):
        # Add the difference of count to the answer
        ans += max(0, count[r][j] - count[l-1][j] - 1)
    print(ans)