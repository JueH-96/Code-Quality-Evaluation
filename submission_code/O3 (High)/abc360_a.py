def main():
    import sys

    S = sys.stdin.readline().strip()
    # Find the positions (0-based indices) of rice and miso soup
    pos_r = S.index('R')
    pos_m = S.index('M')

    # Rice is to the left of miso soup if its index is smaller
    print('Yes' if pos_r < pos_m else 'No')


if __name__ == "__main__":
    main()