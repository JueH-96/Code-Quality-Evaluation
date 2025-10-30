# Read the input
N, X, Y = map(int, input().split())
S = input()
T = input()

# Define a function to perform the operations
def can_transform(S, T, X, Y):
    for i in range(N - (X + Y) + 1):
        # Check if Operation A can be performed
        if S[i:i+X] == '0'*X and S[i+X:i+X+Y] == '1'*Y:
            S = S[:i] + '1'*Y + '0'*X + S[i+X+Y:]
            if S == T:
                return "Yes"
        # Check if Operation B can be performed
        elif S[i:i+Y] == '1'*Y and S[i+Y:i+Y+X] == '0'*X:
            S = S[:i] + '0'*X + '1'*Y + S[i+X+Y:]
            if S == T:
                return "Yes"
    return "No"

# Call the function and print the result
print(can_transform(S, T, X, Y))