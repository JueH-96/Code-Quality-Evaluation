def main():
    import sys
    N = int(sys.stdin.readline().strip())
    s = str(N)
    L = len(s)
    # For numbers with up to 3 digits, we print as is.
    if L <= 3:
        print(N)
    else:
        # Number of least‐significant digits to zero out:
        k = L - 3
        factor = 10 ** k
        approx = (N // factor) * factor
        print(approx)

if __name__ == "__main__":
    main()