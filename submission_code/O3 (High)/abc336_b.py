def main():
    import sys

    # Read the integer N
    N = int(sys.stdin.readline().strip())

    # Isolate the least-significant 1-bit with (N & -N),
    # its bit length minus one equals the number of trailing zeros.
    ctz = (N & -N).bit_length() - 1

    # Output the result
    print(ctz)


# Execute the main function
if __name__ == "__main__":
    main()