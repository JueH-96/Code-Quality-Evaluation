def main():
    import sys

    A = list(map(int, sys.stdin.read().strip().split()))
    target = [1, 2, 3, 4, 5]

    # Already sorted -> need exactly one swap, so it's impossible
    if A == target:
        print("No")
        return

    # Try every adjacent swap
    for i in range(4):  # indices 0..3 (since len==5)
        B = A[:]            # make a copy
        B[i], B[i + 1] = B[i + 1], B[i]  # swap adjacent elements
        if B == target:
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()