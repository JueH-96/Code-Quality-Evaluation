def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # Divide out all factors of 2
    while N % 2 == 0:
        N //= 2
    # Divide out all factors of 3
    while N % 3 == 0:
        N //= 3
    # If we've reduced N to 1, it was of the form 2^x * 3^y
    print("Yes" if N == 1 else "No")

if __name__ == "__main__":
    main()