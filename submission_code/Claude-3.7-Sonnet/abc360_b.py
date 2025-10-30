def check_condition(S, T, c, w):
    # Split S at every w characters
    substrings = [S[i:i+w] for i in range(0, len(S), w)]
    
    # Take the c-th character of each substring if its length is at least c
    result = ''.join([s[c-1] for s in substrings if len(s) >= c])
    
    return result == T

def solve(S, T):
    for w in range(1, len(S)):
        for c in range(1, w + 1):
            if check_condition(S, T, c, w):
                return "Yes"
    return "No"

# Read input
S, T = input().split()

# Print output
print(solve(S, T))