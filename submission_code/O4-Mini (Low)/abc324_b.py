def main():
    import sys
    N = int(sys.stdin.readline().strip())
    # Remove all factors of 2
    while N % 2 == 0:
        N //= 2
    # Remove all factors of 3
    while N % 3 == 0:
        N //= 3
    # If what's left is 1, it's of the form 2^x * 3^y
    print("Yes" if N == 1 else "No")

if __name__ == "__main__":
    main()