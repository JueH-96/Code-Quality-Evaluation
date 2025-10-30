def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    treatments = list(map(int, data[2:]))

    # Initialize all holes with a tooth (True)
    holes = [True] * N

    # Perform the treatments
    for t in treatments:
        holes[t - 1] = not holes[t - 1]  # Toggle the presence of a tooth

    # Count and print the number of holes that still have teeth
    print(sum(holes))

# Do not remove the call to main()
main()