def main():
    import sys
    N = int(sys.stdin.readline().strip())
    # Factor out 2's
    while N % 2 == 0:
        N //= 2
    # Factor out 3's
    while N % 3 == 0:
        N //= 3
    print("Yes" if N == 1 else "No")

# Do not forget to call main
main()