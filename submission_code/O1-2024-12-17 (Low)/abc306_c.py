def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # pos[i] will store the positions (1-based) of element i in the array A
    pos = [[] for _ in range(N+1)]

    # Record the positions of each element in A
    for idx, val in enumerate(A, start=1):
        pos[val].append(idx)

    # Build a list of (middle_position, value)
    middle_positions = []
    for i in range(1, N+1):
        middle_positions.append((pos[i][1], i))  # the 2nd occurrence is the middle one

    # Sort by the middle position
    middle_positions.sort(key=lambda x: x[0])

    # Output the elements in order of their middle occurrence
    print(" ".join(str(pair[1]) for pair in middle_positions))

# Do not remove, this is required to run the solution.
if __name__ == "__main__":
    main()