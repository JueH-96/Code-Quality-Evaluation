import sys

def main():
    # Read all integers from stdin
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:  # Safety check: no input
        return

    N, M = data[0], data[1]
    A = data[2:2 + M]  # Extract the M elements

    present = set(A)
    missing = [i for i in range(1, N + 1) if i not in present]

    # Output
    print(len(missing))
    if missing:
        print(' '.join(map(str, missing)))
    else:
        # When no numbers are missing, print an empty second line
        print()

if __name__ == "__main__":
    main()