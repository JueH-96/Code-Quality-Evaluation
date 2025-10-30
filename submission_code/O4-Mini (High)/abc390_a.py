def main():
    import sys

    A = list(map(int, sys.stdin.read().split()))
    target = [1, 2, 3, 4, 5]

    # Try every possible adjacent swap
    for i in range(4):
        B = A.copy()
        B[i], B[i+1] = B[i+1], B[i]
        if B == target:
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()