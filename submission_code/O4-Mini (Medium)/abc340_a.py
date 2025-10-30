def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, D = map(int, data)
    # Generate arithmetic sequence from A to B inclusive with step D
    seq = range(A, B + 1, D)
    # Convert to strings and print space-separated
    print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()