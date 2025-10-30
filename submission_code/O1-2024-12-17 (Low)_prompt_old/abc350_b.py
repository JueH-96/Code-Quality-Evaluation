def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    T = list(map(int, data[2:]))

    # Initialize an array indicating presence of tooth (True) for each hole
    has_tooth = [True] * (N + 1)  # Index 0 is unused

    for hole in T:
        # Toggle the presence of tooth in the given hole
        has_tooth[hole] = not has_tooth[hole]

    # Count how many holes still have teeth
    print(sum(has_tooth[1:]))

def main():
    solve()

if __name__ == "__main__":
    main()