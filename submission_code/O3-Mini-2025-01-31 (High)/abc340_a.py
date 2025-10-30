def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, D = map(int, data)
    # Generate arithmetic sequence from A to B inclusive using step D.
    sequence = range(A, B + 1, D)  # B + 1 because the end is exclusive.
    # Print the sequence with space separation.
    print(" ".join(map(str, sequence)))

if __name__ == '__main__':
    main()