def main():
    import sys
    A, B, C, D = map(int, sys.stdin.readline().split())
    W = C - A
    H = D - B

    full_x = W // 2
    rem_x = W % 2
    full_y = H // 2
    rem_y = H % 2

    full_area = full_x * full_y * 2  # 2 per 2x2 block

    vertical = (rem_x == 1) * full_y * 1.5
    horizontal = (rem_y == 1) * full_x * 1.5
    small = 0  # small rectangle contributes 0

    total = full_area + vertical + horizontal + small
    print(int(2 * total))

if __name__ == "__main__":
    main()