# Read the input
N, Q = map(int, input().split())
S = input()

# Function to flip the characters in the given range
def flip(s, l, r):
    return s[:l-1] + ''.join('0' if c == '1' else '1' for c in s[l-1:r]) + s[r:]

# Function to check if a substring is a good string
def is_good(s):
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            return "No"
    return "Yes"

# Process the queries
for _ in range(Q):
    query = input().split()
    if query[0] == "1":
        l, r = map(int, query[1:])
        S = flip(S, l, r)
    else:
        l, r = map(int, query[1:])
        print(is_good(S[l-1:r]))