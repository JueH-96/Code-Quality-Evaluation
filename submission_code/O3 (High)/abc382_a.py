def main():
    import sys

    # Read input
    N_D, S = sys.stdin.readline().split(), sys.stdin.readline().strip()
    N, D = map(int, N_D)  # N is not actually needed for the calculation

    # Count boxes that were already empty
    already_empty = S.count('.')

    # After D days, exactly D more boxes (each previously with one cookie) will be empty
    result = already_empty + D

    # Output the answer
    print(result)

# Execute the main function
if __name__ == "__main__":
    main()