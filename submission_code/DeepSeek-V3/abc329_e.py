# YOUR CODE HERE
def can_match(N, M, S, T):
    # Initialize X as a list of '#' characters
    X = ['#'] * N
    # Iterate through S to find positions where T can be placed
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            # Replace the corresponding positions in X with T
            for j in range(M):
                X[i+j] = T[j]
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