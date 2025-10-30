def main():
    import sys
    input_line = sys.stdin.read().strip()
    if not input_line:
        return
    A, B, D = map(int, input_line.split())
    # Generate the arithmetic sequence from A to B with difference D.
    sequence = list(range(A, B + 1, D))
    # Print the sequence with space separation.
    print(" ".join(map(str, sequence)))

if __name__ == '__main__':
    main()