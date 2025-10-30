def solve():
    N, M = map(int, input().split())
    S = input()
    T = input()

    # Initialize X as a string of '#' of length N
    X = '#' * N

    # Iterate over the string S
    for i in range(N):
        # If the character in S is not '#'
        if S[i] != '#':
            # Replace the corresponding character in X with the character in S
            X = X[:i] + S[i] + X[i+1:]

    # Initialize a flag to indicate whether it is possible to make X match S
    possible = True

    # Iterate over the string X
    for i in range(N):
        # If the character in X is not '#' and it is not the same as the corresponding character in S
        if X[i] != '#' and X[i] != S[i]:
            # If the character in X is not the same as the character in T
            if i+M <= N and X[i:i+M] == T:
                # Replace the M consecutive characters in X with T
                X = X[:i] + T + X[i+M:]
            else:
                # If it is not possible to replace the M consecutive characters in X with T
                possible = False
                break

    # If it is possible to make X match S
    if possible:
        print('Yes')
    else:
        print('No')

# Call the function
solve()