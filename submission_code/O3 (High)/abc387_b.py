def main() -> None:
    import sys

    X = int(sys.stdin.readline().strip())

    # Sum of all 81 numbers in the 9×9 multiplication table.
    total = 45 * 45          # (1+2+...+9)^2 = 2025

    # Count how many times X appears in the table.
    occ = 0
    for i in range(1, 10):   # rows (1..9)
        if X % i == 0:
            j = X // i       # corresponding column value
            if 1 <= j <= 9:  # must also be within 1..9
                occ += 1

    # Subtract the contribution of all cells equal to X.
    answer = total - X * occ
    print(answer)


if __name__ == "__main__":
    main()