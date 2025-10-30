def main():
    import sys

    A_str = sys.stdin.readline().strip()
    if not A_str:
        return
    A, B = map(int, A_str.split())

    # Convert each number to (row, col) on a 3x3 board
    # 1 2 3
    # 4 5 6
    # 7 8 9
    def pos(n):
        return (n - 1) // 3, (n - 1) % 3

    r1, c1 = pos(A)
    r2, c2 = pos(B)

    if r1 == r2 and abs(c1 - c2) == 1:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()