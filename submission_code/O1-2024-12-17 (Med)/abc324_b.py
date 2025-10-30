def main():
    N = int(input().strip())
    # Factor out powers of 2
    while N % 2 == 0:
        N //= 2
    # Factor out powers of 3
    while N % 3 == 0:
        N //= 3
    # If we're left with 1, it means N = 2^x * 3^y
    print("Yes" if N == 1 else "No")

# Do not forget to call main()
main()