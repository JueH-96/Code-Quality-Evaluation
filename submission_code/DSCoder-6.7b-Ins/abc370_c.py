def solve():
    S = input().strip()
    T = input().strip()

    # Initialize variables
    X = []
    operations = 0

    # While S is not equal to T
    while S != T:
        # Append S to X
        X.append(S)
        # Increment operations
        operations += 1
        # Find the position where S and T are different
        for i in range(len(S)):
            if S[i] != T[i]:
                # Change the character at position i in S to the character at position i in T
                S = S[:i] + T[i] + S[i+1:]
                break

    # Print the number of operations and the strings in X
    print(operations)
    for string in X:
        print(string)

# Call the function
solve()