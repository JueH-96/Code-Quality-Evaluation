def solve():
    # Read the length of the string
    N = int(input())

    # Read the string
    S = list(input())

    # Read the number of operations
    Q = int(input())

    # Perform the operations
    for _ in range(Q):
        # Read the characters to replace
        c, d = input().split()

        # Replace all occurrences of c with d in the string
        S = [d if char == c else char for char in S]

    # Print the final string
    print(''.join(S))

# Call the function to solve the problem
solve()