def main():
    import sys
    data = sys.stdin.read().split()
    N, Q = map(int, data[:2])
    T = list(map(int, data[2:]))

    # Initialize an array to represent the presence of teeth.
    # True at index i means there's a tooth in hole i.
    has_tooth = [True] * (N + 1)  # we use 1-based indexing; index 0 is unused.

    # Process each treatment: toggle the presence of a tooth in hole T_i.
    for t in T:
        has_tooth[t] = not has_tooth[t]

    # Count how many holes still have a tooth.
    result = sum(has_tooth[1:])  # ignore index 0

    # Print the result.
    print(result)

if __name__ == "__main__":
    main()