def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    total_A = 0
    max_bonus = 0

    # Read each giant's A and B, summing all A and tracking the best (B - A)
    index = 1
    for _ in range(n):
        A = int(data[index])
        B = int(data[index+1])
        index += 2

        total_A += A
        bonus = B - A
        if bonus > max_bonus:
            max_bonus = bonus

    # The maximum head height is the fixed sum of all A's plus
    # the maximum extra height (B - A) from the topmost giant.
    print(total_A + max_bonus)

if __name__ == '__main__':
    main()