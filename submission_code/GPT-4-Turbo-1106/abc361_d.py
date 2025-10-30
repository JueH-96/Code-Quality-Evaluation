def min_operations(N, S, T):
    # Count the number of white and black stones in S and T
    white_S = S.count('W')
    white_T = T.count('W')
    black_S = S.count('B')
    black_T = T.count('B')

    # If the number of white or black stones in S and T are different, it's impossible
    if white_S != white_T or black_S != black_T:
        return -1

    # Initialize the number of operations
    operations = 0

    # Initialize pointers for S and T
    pointer_S = 0
    pointer_T = 0

    # Loop through the stones in S and T
    while pointer_S < N and pointer_T < N:
        # If the stones at the current pointers are the same, move both pointers
        if S[pointer_S] == T[pointer_T]:
            pointer_S += 1
            pointer_T += 1
        # If the stones are different, find the next stone in S that matches the stone at pointer_T in T
        else:
            while pointer_S < N and S[pointer_S] != T[pointer_T]:
                pointer_S += 1
            # If we found a matching stone, increment the operations
            if pointer_S < N:
                operations += 1
            # Move the pointer_S back to the position after the last matching stone
            pointer_S = pointer_T + 1
            # Move the pointer_T forward
            pointer_T += 1

    return operations

# Read input
N = int(input().strip())
S = input().strip()
T = input().strip()

# Calculate and print the result
print(min_operations(N, S, T))