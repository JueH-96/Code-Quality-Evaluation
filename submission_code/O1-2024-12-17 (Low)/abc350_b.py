def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    T = list(map(int, data[2:]))

    # Each hole has initially a tooth, so we can use a list of booleans
    has_tooth = [True] * N

    for hole in T:
        # Convert hole index (1-based) to 0-based
        idx = hole - 1
        # Flip the state: if True -> remove tooth (set to False), if False -> grow tooth (set to True)
        has_tooth[idx] = not has_tooth[idx]

    # Count how many holes still have teeth
    print(sum(has_tooth))

# Do not forget to call main()
main()