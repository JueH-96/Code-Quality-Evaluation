def can_match(N, M, S, T):
    # Initialize X as a list of '#' characters
    X = ['#'] * N
    # Iterate through all possible starting positions for the replacement
    for i in range(N - M + 1):
        # Check if the substring in S matches T
        if S[i:i+M] == T:
            # Replace the corresponding part in X with T
            X[i:i+M] = list(T)
    # Check if X matches S
    return ''.join(X) == S

# Read input
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Determine if it's possible to make X match S
if can_match(N, M, S, T):
    print("Yes")
else:
    print("No")