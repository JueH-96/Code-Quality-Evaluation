def can_match(S, T, N, M):
    for i in range(N - M + 1):
        if S[i:i + M] == T:
            # Replace the matched part with '#'
            S = S[:i] + '#' * M + S[i + M:]
    # If there are no characters left other than '#', it's a match
    return set(S) == {'#'}

# Read input from stdin
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Check if it's possible to make X match S
if can_match(S, T, N, M):
    print("Yes")
else:
    print("No")