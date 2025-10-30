def main():
    import sys

    S = sys.stdin.readline().strip()  # Read the 3-character string.
    a = int(S[0])  # First digit
    b = int(S[2])  # Third digit
    print(a * b)   # Output their product


if __name__ == "__main__":
    main()