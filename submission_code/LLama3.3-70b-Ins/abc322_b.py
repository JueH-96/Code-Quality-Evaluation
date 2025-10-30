def solve():
    # Read input from stdin
    N, M = map(int, input().split())
    S = input()
    T = input()

    # Check if S is a prefix of T
    is_prefix = T[:N] == S

    # Check if S is a suffix of T
    is_suffix = T[-N:] == S

    # Print the answer according to the problem statement
    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

# Call the solve function
solve()